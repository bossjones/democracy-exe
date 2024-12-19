"""Utilities for downloading Twitter content."""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import pathlib
import sys
import tempfile
import traceback

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
        logger.info(f"work_dir: {work_dir}")

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
            # import bpdb
            # bpdb.set_trace()
            files = [str(p) for p in tree_list]
            media_files = filter_media(files)

            logger.info(f"tree_list: {tree_list}")
            logger.info(f"files: {files}")
            logger.info(f"media_files: {media_files}")

            # import bpdb
            # bpdb.set_trace()

            # Parse metadata from info.json
            metadata = _parse_tweet_metadata(work_dir)

            logger.info(f"metadata: {metadata}")

            return DownloadResult(
                success=True,
                metadata=metadata,
                local_files=media_files,
                error=None
            )

        except Exception as e:
            logger.exception(f"Error downloading tweet: {e}")
            print(f"{e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f"Error Class: {e.__class__}")
            output = f"[UNEXPECTED] {type(e).__name__}: {e}"
            print(output)
            print(f"exc_type: {exc_type}")
            print(f"exc_value: {exc_value}")
            traceback.print_tb(exc_traceback)
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
    def find_info_json(directory: pathlib.Path) -> pathlib.Path | None:
        """Recursively search for info.json file.

        Args:
            directory: Starting directory for search

        Returns:
            Path to info.json if found, None otherwise
        """
        logger.info(f"Searching for info.json in: {directory}")

        # Check if directory exists
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return None

        # Walk through directory tree
        for root, dirs, files in os.walk(directory):
            logger.debug(f"Checking directory: {root}")
            logger.debug(f"Found files: {files}")

            if 'info.json' in files:
                info_path = pathlib.Path(root) / 'info.json'
                logger.info(f"Found info.json at: {info_path}")
                return info_path

        logger.warning(f"No info.json found in {directory} or its subdirectories")
        return None

    info_path = find_info_json(pathlib.Path(work_dir))

    if info_path is None or not info_path.exists():
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

    except (json.decoder.JSONDecodeError, FileNotFoundError, KeyError) as e:
        logger.exception(f"Error parsing tweet metadata: {e}")
        return {
            "id": "",
            "url": "",
            "author": "",
            "content": "",
            "media_urls": [],
            "created_at": "",
        }
