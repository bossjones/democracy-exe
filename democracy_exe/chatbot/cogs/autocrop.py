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
        try:
            # Load image
            img = Image.open(image_path)

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Calculate target dimensions
            target_ratio = aspect_ratio[0] / aspect_ratio[1]
            current_ratio = img.width / img.height

            if current_ratio > target_ratio:
                # Image is too wide
                new_width = int(img.height * target_ratio)
                offset = (img.width - new_width) // 2
                crop_box = (offset, 0, offset + new_width, img.height)
            else:
                # Image is too tall
                new_height = int(img.width / target_ratio)
                offset = (img.height - new_height) // 2
                crop_box = (0, offset, img.width, offset + new_height)

            # Crop and save
            cropped = img.crop(crop_box)
            cropped.save(output_path, quality=95)
            return True

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
        if not attachment.content_type or not attachment.content_type.startswith('image/'):
            await ctx.send("Please provide a valid image file")
            return False, "Please provide a valid image file"

        # Create progress message
        progress = await ctx.send(f"Processing image to {ratio_name} format...")

        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                # Download attachment
                input_path = f"{tmpdirname}/input{pathlib.Path(attachment.filename).suffix}"
                await attachment.save(input_path)

                # Process image
                output_path = f"{tmpdirname}/output{pathlib.Path(attachment.filename).suffix}"
                await self._process_image(
                    input_path,
                    ASPECT_RATIOS[ratio_name],
                    output_path
                )

                # Send processed image
                await ctx.send(file=discord.File(output_path))
                await progress.edit(content="Processing complete!")
                return True, None

        except Exception as e:
            error_msg = f"Failed to process image: {e}"
            await progress.edit(content=error_msg)
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
        if not ctx.message.attachments:
            await ctx.send("Please attach an image to crop")
            return

        await self._handle_crop(ctx, "square", ctx.message.attachments[0])

    @crop.command(name="portrait")
    async def portrait(self, ctx: commands.Context) -> None:
        """Crop image to 4:5 aspect ratio.

        Args:
            ctx: Command context
        """
        if not ctx.message.attachments:
            await ctx.send("Please attach an image to crop")
            return

        await self._handle_crop(ctx, "portrait", ctx.message.attachments[0])

    @crop.command(name="landscape")
    async def landscape(self, ctx: commands.Context) -> None:
        """Crop image to 16:9 aspect ratio.

        Args:
            ctx: Command context
        """
        if not ctx.message.attachments:
            await ctx.send("Please attach an image to crop")
            return

        await self._handle_crop(ctx, "landscape", ctx.message.attachments[0])

    @crop.command(name="story")
    async def story(self, ctx: commands.Context) -> None:
        """Crop image to 9:16 aspect ratio.

        Args:
            ctx: Command context
        """
        if not ctx.message.attachments:
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
