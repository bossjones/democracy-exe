"""Test utilities for Discord-related tests."""

from __future__ import annotations

import asyncio

from typing import Any

import aiofiles
import discord

from democracy_exe.aio_settings import aiosettings


class SlowAttachment(discord.Attachment):
    """Mock attachment that simulates a slow download.

    This class is used for testing timeout handling in file operations.
    """

    async def save(self, fp: str, **kwargs: Any) -> None:
        """Simulate a slow file save operation.

        Args:
            fp: File path to save to
            kwargs: Additional arguments passed to save
        """
        await asyncio.sleep(aiosettings.autocrop_download_timeout + 1)
        # Simulate saving the file
        async with aiofiles.open(fp, mode="wb") as f:
            await f.write(b"test content")
