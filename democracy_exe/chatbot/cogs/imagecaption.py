# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

"""Image captioning cog for Discord bot.

This cog provides image captioning functionality using the BLIP model.
It can process images from URLs, attachments, and Tenor GIFs.
"""
from __future__ import annotations

import re

from io import BytesIO
from re import Pattern
from typing import TYPE_CHECKING, Optional, Union
from urllib.parse import urlparse

import discord
import requests
import structlog
import torch

from discord.ext import commands
from discord.ext.commands import BucketType, cooldown


logger = structlog.get_logger(__name__)
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor  # type: ignore

from democracy_exe.factories import guild_factory


if TYPE_CHECKING:
    from discord import Message
    from PIL.Image import Image as PILImage

# URL validation patterns
URL_PATTERN: Pattern = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
IMAGE_EXTENSIONS: tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.gif', '.webp')


class ImageCaptionCog(commands.Cog, name="image_caption"):
    """Cog for handling image captioning using BLIP model.

    This cog provides commands for generating captions for images shared in Discord,
    whether they are direct attachments, URLs, or Tenor GIFs.

    Attributes:
        bot: The Discord bot instance
        processor: BLIP image processor
        model: BLIP captioning model
    """

    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the image captioning cog.

        Args:
            bot: The Discord bot instance
        """
        self.bot = bot
        logger.info("Initializing BLIP model and processor...")

        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                torch_dtype=torch.float32
            ).to("cpu")
            logger.info("Successfully initialized BLIP model and processor")
        except Exception as e:
            logger.error(f"Failed to initialize BLIP model: {e!s}")
            raise

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Handle cog ready event."""
        logger.info(f"{type(self).__name__} Cog ready.")
        # await logger.complete()

    @commands.Cog.listener()
    async def on_guild_join(self, guild: discord.Guild) -> None:
        """Handle guild join event.

        Args:
            guild: The guild that was joined
        """
        logger.info(f"Joined new guild: {guild.id}")
        guild_obj = guild_factory.Guild(id=guild.id)
        # await logger.complete()

    def _validate_image_url(self, url: str) -> bool:
        """Validate if a URL points to an image.

        Args:
            url: URL to validate

        Returns:
            bool: True if URL is valid image URL, False otherwise
        """
        try:
            # Check URL format
            if not URL_PATTERN.match(url):
                return False

            # Parse URL and check extension
            parsed = urlparse(url)
            return parsed.path.lower().endswith(IMAGE_EXTENSIONS)
        except Exception as e:
            logger.error(f"Error validating URL {url}: {e!s}")
            return False

    async def _process_tenor_gif(self, message: discord.Message, url: str) -> str:
        """Process a Tenor GIF URL to extract descriptive text.

        Args:
            message: The Discord message
            url: The Tenor GIF URL

        Returns:
            Processed message content with GIF description
        """
        try:
            # Extract the relevant part of the URL
            start_index = url.index("https://tenor.com/view/")
            end_index = url.find(" ", start_index)
            tenor_url = url[start_index:] if end_index == -1 else url[start_index:end_index]

            # Extract descriptive words from URL
            words = tenor_url.split("/")[-1].split("-")[:-1]
            description = " ".join(words)

            return f"{message.content} [{message.author.display_name} posts an animated {description}]".replace(tenor_url, "") # type: ignore
        except Exception as e:
            logger.error(f"Error processing Tenor GIF: {e!s}")
            # await logger.complete()
            return message.content # type: ignore

    async def _download_image(self, url: str) -> PILImage | None:
        """Download and process an image from a URL.

        Args:
            url: The image URL

        Returns:
            PIL Image object or None if download fails

        Raises:
            requests.RequestException: If image download fails
        """
        try:
            # Validate URL before downloading
            if not self._validate_image_url(url):
                logger.warning(f"Invalid image URL: {url}")
                return None

            async with self.bot.session.get(url) as response: # type: ignore
                if response.status != 200:
                    raise requests.RequestException(f"Failed to download image: {response.status}")
                data = await response.read()
                return Image.open(BytesIO(data)).convert("RGB")
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e!s}")
            # await logger.complete()
            return None

    def caption_image(self, image: PILImage) -> str:
        """Generate a caption for an image using BLIP model.

        Args:
            image: PIL Image to caption

        Returns:
            Generated caption text

        Raises:
            Exception: If caption generation fails
        """
        try:
            inputs = self.processor(image.convert("RGB"), return_tensors="pt").to("cpu", torch.float32)
            out = self.model.generate(**inputs, max_new_tokens=50)
            return self.processor.decode(out[0], skip_special_tokens=True) # type: ignore
        except Exception as e:
            logger.error(f"Error generating caption: {e!s}")
            raise

    @commands.command(
        name="caption",
        aliases=["cap", "describe"],
        help="Generate a caption for an image from URL or attachment"
    )
    @commands.cooldown(1, 30, BucketType.user)  # One use per user every 30 seconds
    async def image_caption(self, ctx: commands.Context, url: str | None = None) -> None:
        """Generate a caption for an image.

        This command can process:
        - Image attachments
        - Image URLs
        - Tenor GIFs

        Args:
            ctx: Command context
            url: Optional image URL
        """

        try:
            # Handle message with no image
            if not url and not ctx.message.attachments: # type: ignore
                await ctx.send("Please provide an image URL or attachment!")
                return

            # Process URL if provided
            if url:
                if "tenor.com/view/" in url:
                    response = await self._process_tenor_gif(ctx.message, url) # type: ignore
                    await ctx.send(response)
                    return

                # Validate URL
                if not self._validate_image_url(url):
                    await ctx.send("Invalid image URL! Please provide a valid image URL.")
                    return

                image = await self._download_image(url)
                if not image:
                    await ctx.send("Failed to download image from URL!")
                    return

            # Process attachment if present
            elif ctx.message.attachments: # type: ignore
                attachment = ctx.message.attachments[0] # type: ignore
                if not attachment.content_type or not attachment.content_type.startswith("image/"):
                    await ctx.send("The attachment must be an image!")
                    return

                image = await self._download_image(attachment.url)
                if not image:
                    await ctx.send("Failed to process image attachment!")
                    return

            # Generate and send caption if image was loaded
            if image:
                try:
                    caption = self.caption_image(image)
                    await ctx.send(f"I see {caption}")
                finally:
                    try:
                        image.close()
                    except Exception as e:
                        logger.warning(f"Error closing image: {e!s}")
            else:
                await ctx.send("Failed to process image!")

        except commands.CommandOnCooldown as e:
            await ctx.send(f"This command is on cooldown. Try again in {e.retry_after:.1f} seconds.") # type: ignore
        except Exception as e:
            logger.exception("Error in image_caption command")
            # await logger.complete()
            await ctx.send(f"An error occurred while processing the image: {e!s}")


async def setup(bot: commands.Bot) -> None:
    """Add ImageCaptionCog to bot.

    Args:
        bot: The Discord bot instance
    """
    await bot.add_cog(ImageCaptionCog(bot))
