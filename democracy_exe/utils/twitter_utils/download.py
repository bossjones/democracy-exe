"""Utilities for downloading Twitter content."""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import tempfile

from typing import Any, Dict, List, Optional

import rich

from loguru import logger
from tqdm.auto import tqdm

from democracy_exe import shell
from democracy_exe.constants import DL_SAFE_TWITTER_COMMAND, DL_TWITTER_CARD_COMMAND, DL_TWITTER_THREAD_COMMAND
from democracy_exe.utils.file_functions import filter_media, tree

from .types import DownloadResult, TweetDownloadMode, TweetMetadata


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
    with tempfile.TemporaryDirectory() as tmpdirname:
        work_dir = working_dir or tmpdirname

        try:
            # Select command based on mode
            cmd = {
                "single": DL_SAFE_TWITTER_COMMAND,
                "thread": DL_TWITTER_THREAD_COMMAND,
                "card": DL_TWITTER_CARD_COMMAND,
            }[mode]

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
    """Parse tweet metadata from gallery-dl info.json file."""
    # TODO: Implement metadata parsing
    return {
        "id": "",
        "url": "",
        "author": "",
        "content": "",
        "media_urls": [],
        "created_at": "",
    }
