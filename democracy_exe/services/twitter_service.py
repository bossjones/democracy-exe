"""Twitter service for handling Twitter-related operations."""
from __future__ import annotations

import asyncio
import pathlib
import tempfile

from typing import List, Optional, Tuple

import aiohttp
import discord

from loguru import logger

from democracy_exe.services.base_service import BaseService
from democracy_exe.utils import file_functions
from democracy_exe.utils.events import aio_create_thumbnail_attachment


class TwitterService(BaseService):
    """Service for handling Twitter operations."""

    async def download_media(self, url: str, download_dir: str) -> tuple[list[str], list[str]]:
        """Download media from Twitter URL.

        Args:
            url: Twitter URL to download from
            download_dir: Directory to download files to

        Returns:
            Tuple of (media files, json files)
        """
        # Download logic here
        pass

    async def process_attachments(
        self,
        message: discord.Message,
        download_dir: str
    ) -> tuple[list[str], list[str]]:
        """Process message attachments.

        Args:
            message: Discord message with attachments
            download_dir: Directory to save attachments to

        Returns:
            Tuple of (media files, json files)
        """
        media_files = []
        json_files = []

        for attachment in message.attachments:
            file_path = pathlib.Path(download_dir) / attachment.filename
            await attachment.save(file_path)

            if str(file_path) in file_functions.filter_media([str(file_path)]):
                media_files.append(str(file_path))
            elif file_path.suffix.lower() in file_functions.JSON_EXTENSIONS:
                json_files.append(str(file_path))

        return media_files, json_files

    async def create_embed(
        self,
        url: str,
        channel: discord.TextChannel,
        json_data: dict,
        is_dropbox: bool = False
    ) -> discord.Embed:
        """Create Twitter embed message.

        Args:
            url: Original Twitter URL
            channel: Discord channel
            json_data: Tweet metadata
            is_dropbox: Whether content was uploaded to Dropbox

        Returns:
            Discord embed object
        """
        # Embed creation logic here
        pass
