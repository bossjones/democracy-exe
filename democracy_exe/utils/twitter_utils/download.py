"""Utilities for downloading Twitter content."""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import pathlib
import tempfile

from typing import Any, Dict, List, Optional

import aiohttp
import rich

from loguru import logger
from tqdm.auto import tqdm

from democracy_exe import shell
from democracy_exe.constants import DL_SAFE_TWITTER_COMMAND, DL_TWITTER_CARD_COMMAND, DL_TWITTER_THREAD_COMMAND
from democracy_exe.utils.file_functions import filter_media, tree

from .types import DownloadResult, TweetDownloadMode, TweetMetadata


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


async def download_tweet(
    url: str,
    mode: TweetDownloadMode = "single",
    working_dir: str | None = None,
) -> DownloadResult:
    """Download content from a tweet URL.

    Args:
        url: The tweet URL to download from
        mode: The download mode - single tweet, thread, or card
        working_dir: Optional working directory for downloads

    Returns:
        DownloadResult containing success status, metadata and local files
    """
    # Use temporary dir if none provided
    with tempfile.TemporaryDirectory(delete=False) as tmpdirname:
        work_dir = working_dir or tmpdirname

        try:
            # Select command based on mode
            cmd = {
                "single": DL_SAFE_TWITTER_COMMAND,
                "thread": DL_TWITTER_THREAD_COMMAND,
                "card": DL_TWITTER_CARD_COMMAND,
            }[mode]

            logger.info(f"cmd: {cmd}")
            logger.info(f"url: {url}")
            logger.info(f"work_dir: {work_dir}")

            # Execute download command
            await shell._aio_run_process_and_communicate(
                cmd.format(dl_uri=url).split(),
                cwd=work_dir
            )

            # Get downloaded files
            tree_list = tree(pathlib.Path(work_dir))
            files = [str(p) for p in tree_list]
            media_files = filter_media(files)

            # Parse metadata from info.json
            metadata = _parse_tweet_metadata(work_dir)

            return DownloadResult(
                success=True,
                metadata=metadata,
                local_files=media_files,
                error=None
            )

        except Exception as e:
            logger.exception(f"Error downloading tweet: {e}")
            return DownloadResult(
                success=False,
                metadata={
                    "id": "",
                    "url": url,
                    "author": "",
                    "content": "",
                    "media_urls": [],
                    "created_at": "",
                },
                local_files=[],
                error=str(e)
            )


def _parse_tweet_metadata(work_dir: str) -> TweetMetadata:
    """
    Recursively search for and parse tweet metadata from gallery-dl info.json file.

    Args:
        work_dir: Directory to start searching for info.json file

    Returns:
        Dictionary containing tweet metadata
    """
    def find_info_json(directory: pathlib.Path) -> pathlib.Path:
        for root, _, files in os.walk(directory):
            if 'info.json' in files:
                return pathlib.Path(root) / 'info.json'
        return pathlib.Path()

    info_path = find_info_json(pathlib.Path(work_dir))

    if not info_path.exists():
        logger.warning(f"No info.json found in {work_dir} or its subdirectories")
        return {
            "id": "",
            "url": "",
            "author": "",
            "content": "",
            "media_urls": [],
            "created_at": "",
        }

    try:
        import json
        with open(info_path) as f:
            data = json.load(f)

        # Extract metadata from gallery-dl json format
        tweet_data = data.get("tweet", {})
        user_data = tweet_data.get("user", {})

        media_urls = [media["url"] for media in tweet_data.get("media", []) if "url" in media]

        return {
            "id": str(tweet_data.get("id", "")),
            "url": tweet_data.get("url", ""),
            "author": user_data.get("name", ""),
            "content": tweet_data.get("text", ""),
            "media_urls": media_urls,
            "created_at": tweet_data.get("created_at", ""),
        }

    except Exception as e:
        logger.exception(f"Error parsing tweet metadata: {e}")
        return {
            "id": "",
            "url": "",
            "author": "",
            "content": "",
            "media_urls": [],
            "created_at": "",
        }
