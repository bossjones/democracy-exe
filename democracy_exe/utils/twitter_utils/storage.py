"""Storage utilities for Twitter content."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Final, List, Optional, Set, TypedDict, Union

from loguru import logger

from democracy_exe.utils.twitter_utils.models import (
    DownloadedContent,
    MediaItem,
    MediaType,
    Tweet,
    TweetCard,
    TweetThread,
)


# Constants
MAX_CACHE_AGE: Final[timedelta] = timedelta(hours=24)
MAX_CACHE_SIZE: Final[int] = 1024 * 1024 * 1024  # 1GB
CHUNK_SIZE: Final[int] = 8192  # 8KB chunks for file operations


class StorageStats(TypedDict):
    """Type hints for storage statistics."""

    total_size: int
    file_count: int
    oldest_file: datetime | None
    newest_file: datetime | None


class StorageError(Exception):
    """Base exception for storage-related errors."""


class FileNotFoundError(StorageError):
    """Raised when a file is not found."""


class StorageFullError(StorageError):
    """Raised when storage is full."""


class TwitterMediaStorage:
    """Handles storage of Twitter media files.

    This class manages both temporary and persistent storage of downloaded
    media files, including cleanup of old files and directory structure.

    Attributes:
        base_dir: Base directory for media storage
        temp_dir: Directory for temporary files
        max_cache_age: Maximum age of cached files
        max_cache_size: Maximum size of cache in bytes
    """

    def __init__(
        self,
        base_dir: str | Path | None = None,
        max_cache_age: timedelta = MAX_CACHE_AGE,
        max_cache_size: int = MAX_CACHE_SIZE,
    ) -> None:
        """Initialize storage manager.

        Args:
            base_dir: Base directory for media storage
            max_cache_age: Maximum age of cached files
            max_cache_size: Maximum size of cache in bytes
        """
        if base_dir is None:
            base_dir = Path.home() / ".democracy_exe" / "twitter_media"
        self.base_dir = Path(base_dir)
        self.temp_dir = self.base_dir / "temp"
        self.max_cache_age = max_cache_age
        self.max_cache_size = max_cache_size

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def get_storage_stats(self) -> StorageStats:
        """Get storage statistics.

        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        file_count = 0
        oldest_time: float | None = None
        newest_time: float | None = None

        for path in self.temp_dir.rglob("*"):
            if path.is_file():
                stat = path.stat()
                total_size += stat.st_size
                file_count += 1
                mtime = stat.st_mtime

                if oldest_time is None or mtime < oldest_time:
                    oldest_time = mtime
                if newest_time is None or mtime > newest_time:
                    newest_time = mtime

        return StorageStats(
            total_size=total_size,
            file_count=file_count,
            oldest_file=datetime.fromtimestamp(oldest_time) if oldest_time else None,
            newest_file=datetime.fromtimestamp(newest_time) if newest_time else None,
        )

    async def validate_storage(self) -> None:
        """Validate storage state and clean up if needed.

        Raises:
            StorageFullError: If storage is full and cleanup fails
        """
        stats = await self.get_storage_stats()
        if stats["total_size"] > self.max_cache_size:
            await self.cleanup_old_files()
            # Check if cleanup was sufficient
            new_stats = await self.get_storage_stats()
            if new_stats["total_size"] > self.max_cache_size:
                raise StorageFullError("Storage is full after cleanup")

    async def cleanup_old_files(self) -> None:
        """Remove old files from cache."""
        cutoff = datetime.now() - self.max_cache_age
        total_size = 0
        files_by_time: list[tuple[float, Path]] = []

        # Collect file information
        for path in self.temp_dir.rglob("*"):
            if path.is_file():
                try:
                    mtime = path.stat().st_mtime
                    size = path.stat().st_size
                    total_size += size
                    files_by_time.append((mtime, path))
                except Exception as e:
                    logger.warning(f"Failed to stat {path}: {e}")

        # Sort by modification time
        files_by_time.sort()

        # Remove old files and enforce size limit
        for mtime, path in files_by_time:
            try:
                if (
                    datetime.fromtimestamp(mtime) < cutoff
                    or total_size > self.max_cache_size
                ):
                    size = path.stat().st_size
                    path.unlink()
                    total_size -= size
                    logger.debug(f"Removed old file: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove {path}: {e}")

    def _get_media_dir(self, tweet_id: str) -> Path:
        """Get directory for storing media from a tweet.

        Args:
            tweet_id: ID of the tweet

        Returns:
            Path to media directory
        """
        return self.base_dir / tweet_id

    async def _copy_file_with_progress(
        self,
        src: Path,
        dst: Path,
        *,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        """Copy file with progress tracking.

        Args:
            src: Source file path
            dst: Destination file path
            chunk_size: Size of chunks to copy

        Raises:
            StorageError: If copy fails
        """
        try:
            total_size = src.stat().st_size
            copied = 0

            with src.open("rb") as fsrc, dst.open("wb") as fdst:
                while True:
                    chunk = fsrc.read(chunk_size)
                    if not chunk:
                        break
                    fdst.write(chunk)
                    copied += len(chunk)
                    logger.debug(f"Copied {copied}/{total_size} bytes")

        except Exception as e:
            # Clean up partial file
            if dst.exists():
                dst.unlink()
            raise StorageError(f"Failed to copy file: {e}") from e

    async def save_media_item(
        self,
        media: MediaItem,
        tweet_id: str,
        *,
        temp: bool = True
    ) -> Path:
        """Save a media item to storage.

        Args:
            media: Media item to save
            tweet_id: ID of the tweet
            temp: Whether to use temporary storage

        Returns:
            Path where media was saved

        Raises:
            StorageError: If saving fails
            FileNotFoundError: If media file not found
        """
        if not media.local_path or not media.local_path.exists():
            raise FileNotFoundError(f"Media file not found: {media.local_path}")

        try:
            # Validate storage before saving
            await self.validate_storage()

            if temp:
                # Use temporary directory with tweet ID subdirectory
                target_dir = self.temp_dir / tweet_id
            else:
                target_dir = self._get_media_dir(tweet_id)

            target_dir.mkdir(parents=True, exist_ok=True)

            # Generate target filename
            suffix = media.local_path.suffix
            filename = f"{media.type.value}_{media.local_path.stem}{suffix}"
            target_path = target_dir / filename

            # Copy file with progress
            await self._copy_file_with_progress(media.local_path, target_path)
            media.local_path = target_path

            return target_path

        except Exception as e:
            raise StorageError(f"Failed to save media: {e}") from e

    async def save_tweet_content(
        self,
        content: DownloadedContent,
        *,
        temp: bool = True
    ) -> DownloadedContent:
        """Save all media from downloaded content.

        Args:
            content: Downloaded content to save
            temp: Whether to use temporary storage

        Returns:
            Updated content with new file paths

        Raises:
            StorageError: If saving fails
        """
        if not content.content:
            return content

        try:
            # Clean up old files first
            await self.cleanup_old_files()

            # Handle single tweet
            if isinstance(content.content, Tweet):
                tweet = content.content
                content.local_files = []

                # Save media items
                for media in tweet.media:
                    if media.local_path:
                        path = await self.save_media_item(
                            media,
                            tweet.id,
                            temp=temp
                        )
                        content.local_files.append(path)

                # Save card image if present
                if tweet.card and tweet.card.local_image:
                    card_path = await self.save_media_item(
                        MediaItem(
                            url=tweet.card.image_url or "",
                            type=MediaType.IMAGE,
                            local_path=tweet.card.local_image
                        ),
                        tweet.id,
                        temp=temp
                    )
                    tweet.card.local_image = card_path
                    content.local_files.append(card_path)

            # Handle thread
            elif isinstance(content.content, TweetThread):
                content.local_files = []
                for tweet in content.content.tweets:
                    # Save each tweet's media
                    for media in tweet.media:
                        if media.local_path:
                            path = await self.save_media_item(
                                media,
                                tweet.id,
                                temp=temp
                            )
                            content.local_files.append(path)

            return content

        except Exception as e:
            raise StorageError(f"Failed to save content: {e}") from e

    async def cleanup(self) -> None:
        """Clean up storage directories."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir.mkdir(parents=True)
        except Exception as e:
            logger.warning(f"Failed to clean up storage: {e}")


async def get_storage() -> TwitterMediaStorage:
    """Get storage manager instance.

    Returns:
        Storage manager instance
    """
    return TwitterMediaStorage()
