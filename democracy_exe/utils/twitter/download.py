"""Twitter download utilities."""
from __future__ import annotations

import asyncio
import pathlib

from typing import List, Optional

import aiohttp

from loguru import logger


async def download_media(url: str, download_dir: str) -> list[str]:
    """Download media from Twitter URL.

    Args:
        url: Twitter URL
        download_dir: Directory to save files

    Returns:
        List of downloaded file paths
    """
    # Download implementation
    pass

async def download_thumbnail(url: str, path: str) -> str | None:
    """Download thumbnail image.

    Args:
        url: Image URL
        path: Save path

    Returns:
        Path to downloaded thumbnail or None
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    file_path = pathlib.Path(path)
                    file_path.write_bytes(content)
                    return str(file_path)
    except Exception as e:
        logger.error(f"Error downloading thumbnail: {e}")
        return None
