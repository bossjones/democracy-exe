# pyright: reportAttributeAccessIssue=false

"""Autocrop cog for Discord bot.

This cog provides image auto-cropping functionality including smart detection
of important regions and automatic resizing/cropping to common aspect ratios.

Attributes:
    bot: The Discord bot instance
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import pathlib
import sys
import tempfile
import traceback

from typing import Any, Dict, Final, List, Optional, Tuple, Union

import bpdb
import cv2
import discord
import numpy as np

from discord.ext import commands
from discord.ext.commands import Context
from loguru import logger
from PIL import Image
from rich.pretty import pprint

from democracy_exe.aio_settings import aiosettings
from democracy_exe.factories.guild_factory import Guild
from democracy_exe.utils import file_functions


# Command constants
HELP_MESSAGE: Final[str] = f"""
Available commands:
- {aiosettings.prefix}crop square <attachment>: Crop image to 1:1 aspect ratio
- {aiosettings.prefix}crop portrait <attachment>: Crop image to 4:5 aspect ratio
- {aiosettings.prefix}crop landscape <attachment>: Crop image to 16:9 aspect ratio
- {aiosettings.prefix}crop story <attachment>: Crop image to 9:16 aspect ratio
"""

# Aspect ratio constants
ASPECT_RATIOS = {
    "square": (1, 1),
    "portrait": (4, 5),
    "landscape": (16, 9),
    "story": (9, 16)
}

class AutocropError(Exception):
    """Base exception for autocrop-related errors."""

class Autocrop(commands.Cog):
    """Autocrop functionality for Discord bot.

    Handles intelligent cropping of images to various aspect ratios.

    Attributes:
        bot: The Discord bot instance
    """

    def __init__(self, bot: commands.Bot):
        logger.debug("Initializing Autocrop cog")
        self.bot = bot
        logger.debug("Autocrop cog initialized")

    @commands.Cog.listener()
    async def on_ready(self):
        logger.debug(f"{type(self).__name__} Cog ready.")
        print(f"{type(self).__name__} Cog ready.")
        await logger.complete()

    @commands.Cog.listener()
    async def on_guild_join(self, guild):
        """Add new guilds to the database"""
        logger.debug(f"Adding new guild to database: {guild.id}")
        guild_obj = Guild(id=guild.id)
        logger.debug(f"Successfully added guild {guild.id} to database")

    async def _process_image(
        self,
        image_path: str,
        aspect_ratio: tuple[int, int],
        output_path: str
    ) -> bool:
        """Process image with smart cropping.

        Args:
            image_path: Path to input image
            aspect_ratio: Target width/height ratio tuple
            output_path: Path to save processed image

        Returns:
            bool: Success status

        Raises:
            AutocropError: If processing fails
        """
        logger.debug(f"Starting image processing - Input: {image_path}, Target ratio: {aspect_ratio}")
        try:
            # Run CPU-bound operations in thread pool
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                def process_in_thread():
                    # Load image
                    logger.debug(f"Loading image from {image_path}")
                    img = Image.open(image_path)
                    logger.debug(f"Original image dimensions: {img.size}, mode: {img.mode}")

                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        logger.debug(f"Converting image from {img.mode} to RGB")
                        img = img.convert('RGB')

                    # Calculate target dimensions
                    target_ratio = aspect_ratio[0] / aspect_ratio[1]
                    current_ratio = img.width / img.height
                    logger.debug(f"Target ratio: {target_ratio:.2f}, Current ratio: {current_ratio:.2f}")

                    if current_ratio > target_ratio:
                        # Image is too wide
                        new_width = int(img.height * target_ratio)
                        offset = (img.width - new_width) // 2
                        crop_box = (offset, 0, offset + new_width, img.height)
                        logger.debug(f"Image too wide - New width: {new_width}, Offset: {offset}")
                    else:
                        # Image is too tall
                        new_height = int(img.width / target_ratio)
                        offset = (img.height - new_height) // 2
                        crop_box = (0, offset, img.width, offset + new_height)
                        logger.debug(f"Image too tall - New height: {new_height}, Offset: {offset}")

                    logger.debug(f"Applying crop with box: {crop_box}")
                    # Crop and save
                    cropped = img.crop(crop_box)
                    logger.debug(f"Saving processed image to {output_path} with quality 95")
                    cropped.save(output_path, quality=95)
                    return True

                logger.debug("Dispatching image processing to thread pool")
                return await loop.run_in_executor(pool, process_in_thread)

        except Exception as e:
            logger.exception(f"Failed to process image: {e}")
            raise AutocropError(f"Image processing failed: {e}")

    async def _handle_crop(
        self,
        ctx: Context,
        ratio_name: str,
        attachment: discord.Attachment
    ) -> tuple[bool, str | None]:
        """Handle image cropping workflow.

        Args:
            ctx: Command context
            ratio_name: Name of aspect ratio to use
            attachment: Image attachment to process

        Returns:
            Tuple containing:
                - Success status (bool)
                - Error message if any (Optional[str])
        """
        logger.info(f"Starting crop operation - User: {ctx.author}, Guild: {ctx.guild}, Ratio: {ratio_name}")
        logger.debug(f"Attachment details - Name: {attachment.filename}, Size: {attachment.size}, Type: {attachment.content_type}")

        if not attachment.content_type or not attachment.content_type.startswith('image/'):
            logger.warning(f"Invalid attachment type: {attachment.content_type}")
            await ctx.send("Please provide a valid image file")
            return False, "Please provide a valid image file"

        # Create progress message
        try:
            progress = await ctx.send(f"Processing image to {ratio_name} format...")
            logger.debug("Progress message created successfully")
        except discord.HTTPException as e:
            logger.error(f"Failed to send progress message: {e}")
            return False, "Failed to send progress message"

        try:
            # Use unique temporary directory for each request
            tmp_prefix = f"autocrop_{ctx.message.id}_{ctx.author.id}_"
            logger.debug(f"Creating temporary directory with prefix: {tmp_prefix}")
            with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmpdirname:
                # Download attachment with timeout
                input_path = f"{tmpdirname}/input{pathlib.Path(attachment.filename).suffix}"
                try:
                    logger.debug(f"Downloading attachment with {aiosettings.autocrop_download_timeout}s timeout")
                    async with asyncio.timeout(aiosettings.autocrop_download_timeout):
                        await attachment.save(input_path)
                    logger.debug("Attachment downloaded successfully")
                except TimeoutError:
                    logger.error("Attachment download timed out")
                    await progress.edit(content="Image download timed out")
                    return False, "Image download timed out"
                except discord.HTTPException as e:
                    logger.error(f"Failed to download attachment: {e}")
                    await progress.edit(content=f"Failed to download image: {e}")
                    return False, f"Failed to download image: {e}"

                # Process image
                output_path = f"{tmpdirname}/output{pathlib.Path(attachment.filename).suffix}"
                try:
                    logger.debug(f"Processing image with {aiosettings.autocrop_processing_timeout}s timeout")
                    async with asyncio.timeout(aiosettings.autocrop_processing_timeout):
                        await self._process_image(
                            input_path,
                            ASPECT_RATIOS[ratio_name],
                            output_path
                        )
                    logger.debug("Image processed successfully")
                except TimeoutError:
                    logger.error("Image processing timed out")
                    await progress.edit(content="Image processing timed out")
                    return False, "Image processing timed out"

                # Send processed image
                try:
                    logger.debug(f"Sending processed image: {output_path}")
                    await ctx.send(file=discord.File(output_path))
                    await progress.edit(content="Processing complete!")
                    logger.info(f"Crop operation completed successfully for {ctx.author} in {ctx.guild}")
                    return True, None
                except discord.HTTPException as e:
                    error_msg = f"Failed to send processed image: {e}"
                    logger.error(error_msg)
                    await progress.edit(content=error_msg)
                    return False, error_msg

        except Exception as e:
            error_msg = f"Failed to process image: {e}"
            logger.exception(error_msg)
            try:
                await progress.edit(content=error_msg)
            except discord.HTTPException:
                logger.error(f"Failed to edit progress message: {e}")
            return False, error_msg

    @commands.group(name="crop")
    async def crop(self, ctx: commands.Context) -> None:
        """Autocrop command group."""
        logger.debug(f"Crop command invoked by {ctx.author} in {ctx.guild}")
        if ctx.invoked_subcommand is None:
            logger.debug("No subcommand specified, sending help message")
            await ctx.send(HELP_MESSAGE)

    @crop.command(name="square")
    async def square(self, ctx: commands.Context) -> None:
        """Crop image to 1:1 aspect ratio.

        Args:
            ctx: Command context
        """
        logger.debug(f"Square crop command invoked by {ctx.author}")
        if not ctx.message.attachments:
            logger.warning(f"No attachment provided by {ctx.author}")
            await ctx.send("Please attach an image to crop")
            return

        await self._handle_crop(ctx, "square", ctx.message.attachments[0])

    @crop.command(name="portrait")
    async def portrait(self, ctx: commands.Context) -> None:
        """Crop image to 4:5 aspect ratio.

        Args:
            ctx: Command context
        """
        logger.debug(f"Portrait crop command invoked by {ctx.author}")
        if not ctx.message.attachments:
            logger.warning(f"No attachment provided by {ctx.author}")
            await ctx.send("Please attach an image to crop")
            return

        await self._handle_crop(ctx, "portrait", ctx.message.attachments[0])

    @crop.command(name="landscape")
    async def landscape(self, ctx: commands.Context) -> None:
        """Crop image to 16:9 aspect ratio.

        Args:
            ctx: Command context
        """
        logger.debug(f"Landscape crop command invoked by {ctx.author}")
        if not ctx.message.attachments:
            logger.warning(f"No attachment provided by {ctx.author}")
            await ctx.send("Please attach an image to crop")
            return

        await self._handle_crop(ctx, "landscape", ctx.message.attachments[0])

    @crop.command(name="story")
    async def story(self, ctx: commands.Context) -> None:
        """Crop image to 9:16 aspect ratio.

        Args:
            ctx: Command context
        """
        logger.debug(f"Story crop command invoked by {ctx.author}")
        if not ctx.message.attachments:
            logger.warning(f"No attachment provided by {ctx.author}")
            await ctx.send("Please attach an image to crop")
            return

        await self._handle_crop(ctx, "story", ctx.message.attachments[0])

async def setup(bot: commands.Bot) -> None:
    """Add Autocrop cog to bot.

    Args:
        bot: Discord bot instance
    """
    logger.debug("Setting up Autocrop cog")
    await bot.add_cog(Autocrop(bot))
    logger.debug("Autocrop cog setup complete")
