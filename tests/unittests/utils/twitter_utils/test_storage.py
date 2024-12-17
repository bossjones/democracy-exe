"""Tests for Twitter storage utilities."""

from __future__ import annotations

import json
import shutil

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import pytest

from democracy_exe.utils.twitter_utils.models import (
    DownloadDict,
    DownloadedContent,
    MediaItem,
    MediaType,
    ThreadDict,
    Tweet,
    TweetCard,
    TweetDict,
    TweetThread,
)
from democracy_exe.utils.twitter_utils.storage import (
    FileNotFoundError,
    StorageError,
    StorageFullError,
    TwitterMediaStorage,
)


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


# Test Data
TEST_TWEET_ID = "123456789"
TEST_AUTHOR = "test_user"
TEST_CONTENT = "Test tweet content"
TEST_URL = f"https://twitter.com/{TEST_AUTHOR}/status/{TEST_TWEET_ID}"
TEST_CREATED_AT = datetime(2024, 1, 1, 12, 0, 0)
TEST_MEDIA_URL = "https://example.com/image.jpg"
TEST_CARD_URL = "https://example.com"
TEST_CARD_TITLE = "Test Card"
TEST_CARD_DESCRIPTION = "Card description"
TEST_CARD_IMAGE = "https://example.com/card.jpg"


@pytest.fixture
def storage_root(tmp_path: Path) -> Path:
    """Create a temporary storage root directory.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to storage root
    """
    storage_dir = tmp_path / "twitter_storage"
    storage_dir.mkdir()
    return storage_dir


@pytest.fixture
def tweet_storage(storage_root: Path) -> TwitterMediaStorage:
    """Create a TwitterMediaStorage instance.

    Args:
        storage_root: Storage root directory

    Returns:
        TwitterMediaStorage instance
    """
    return TwitterMediaStorage(storage_root)


@pytest.fixture
def sample_tweet(tmp_path: Path) -> Tweet:
    """Create a sample tweet.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Sample tweet
    """
    media = MediaItem(
        url=TEST_MEDIA_URL,
        type=MediaType.IMAGE,
        local_path=tmp_path / "media.jpg",
    )
    card = TweetCard(
        url=TEST_CARD_URL,
        title=TEST_CARD_TITLE,
        description=TEST_CARD_DESCRIPTION,
        image_url=TEST_CARD_IMAGE,
        local_image=tmp_path / "card.jpg",
    )
    return Tweet(
        id=TEST_TWEET_ID,
        author=TEST_AUTHOR,
        content=TEST_CONTENT,
        created_at=TEST_CREATED_AT,
        url=TEST_URL,
        media=[media],
        card=card,
        retweet_count=10,
        like_count=20,
        reply_count=5,
    )


@pytest.fixture
def sample_thread(sample_tweet: Tweet) -> TweetThread:
    """Create a sample tweet thread.

    Args:
        sample_tweet: Sample tweet

    Returns:
        Sample tweet thread
    """
    tweets = [
        Tweet(
            id=f"{TEST_TWEET_ID}{i}",
            author=TEST_AUTHOR,
            content=f"Tweet {i + 1}",
            created_at=TEST_CREATED_AT,
            url=TEST_URL,
        )
        for i in range(3)
    ]
    return TweetThread(
        tweets=tweets,
        author=TEST_AUTHOR,
        created_at=TEST_CREATED_AT,
    )


class TestTwitterMediaStorage:
    """Tests for TwitterMediaStorage class."""

    def test_init(self, storage_root: Path) -> None:
        """Test TweetStorage initialization.

        Args:
            storage_root: Storage root directory
        """
        storage = TwitterMediaStorage(storage_root)
        assert storage.base_dir == storage_root
        assert storage.base_dir.exists()
        assert storage.base_dir.is_dir()

    def test_init_nonexistent_root(self, tmp_path: Path) -> None:
        """Test TweetStorage initialization with nonexistent root.

        Args:
            tmp_path: Temporary directory path
        """
        storage_dir = tmp_path / "nonexistent"
        storage = TwitterMediaStorage(storage_dir)
        assert storage.base_dir.exists()
        assert storage.base_dir.is_dir()

    def test_init_with_file(self, tmp_path: Path) -> None:
        """Test TwitterMediaStorage initialization with file path.

        Args:
            tmp_path: Temporary directory path
        """
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(FileExistsError):
            TwitterMediaStorage(file_path)

    async def test_get_storage_stats(self, tweet_storage: TwitterMediaStorage) -> None:
        """Test get_storage_stats method.

        Args:
            tweet_storage: TwitterMediaStorage instance
        """
        # Create some test files
        test_file = tweet_storage.temp_dir / "test.txt"
        test_file.write_text("test content")

        stats = await tweet_storage.get_storage_stats()
        assert stats["total_size"] > 0
        assert stats["file_count"] == 1
        assert stats["oldest_file"] is not None
        assert stats["newest_file"] is not None

    async def test_validate_storage(self, tweet_storage: TwitterMediaStorage) -> None:
        """Test validate_storage method.

        Args:
            tweet_storage: TwitterMediaStorage instance
        """
        # Set very small max size
        tweet_storage.max_cache_size = 10  # 10 bytes

        # Create test file much larger than max size (>20 bytes)
        test_file = tweet_storage.temp_dir / "test.txt"
        test_file.write_text("x" * 1000)  # 1KB of data

        with pytest.raises(StorageFullError):
            await tweet_storage.validate_storage()

    async def test_cleanup_old_files(self, tweet_storage: TwitterMediaStorage) -> None:
        """Test cleanup_old_files method.

        Args:
            tweet_storage: TwitterMediaStorage instance
        """
        # Create some test files
        for i in range(3):
            test_file = tweet_storage.temp_dir / f"test{i}.txt"
            test_file.write_text(f"test content {i}")

        await tweet_storage.cleanup_old_files()
        stats = await tweet_storage.get_storage_stats()
        assert stats["file_count"] >= 0

    async def test_save_media_item(self, tweet_storage: TwitterMediaStorage, tmp_path: Path) -> None:
        """Test save_media_item method.

        Args:
            tweet_storage: TwitterMediaStorage instance
            tmp_path: Temporary directory path
        """
        # Create test media file
        media_file = tmp_path / "test.jpg"
        media_file.write_bytes(b"test image content")

        media_item = MediaItem(url="https://example.com/test.jpg", type=MediaType.IMAGE, local_path=media_file)

        saved_path = await tweet_storage.save_media_item(media_item, TEST_TWEET_ID)
        assert saved_path.exists()
        assert saved_path.parent == tweet_storage.temp_dir / TEST_TWEET_ID

    async def test_save_tweet_content(
        self, tweet_storage: TwitterMediaStorage, sample_tweet: Tweet, tmp_path: Path
    ) -> None:
        """Test save_tweet_content method.

        Args:
            tweet_storage: TwitterMediaStorage instance
            sample_tweet: Sample tweet
            tmp_path: Temporary directory path
        """
        # Create test media files
        media_file = tmp_path / "media.jpg"
        media_file.write_bytes(b"test media content")
        card_file = tmp_path / "card.jpg"
        card_file.write_bytes(b"test card content")

        # Update sample tweet with local files
        sample_tweet.media[0].local_path = media_file
        if sample_tweet.card:
            sample_tweet.card.local_image = card_file

        content = DownloadedContent(mode="single", content=sample_tweet, local_files=[])

        saved_content = await tweet_storage.save_tweet_content(content)
        assert len(saved_content.local_files) > 0
        assert all(Path(f).exists() for f in saved_content.local_files)

    async def test_cleanup(self, tweet_storage: TwitterMediaStorage) -> None:
        """Test cleanup method.

        Args:
            tweet_storage: TwitterMediaStorage instance
        """
        # Create some test files
        test_file = tweet_storage.temp_dir / "test.txt"
        test_file.write_text("test content")

        await tweet_storage.cleanup()
        assert not test_file.exists()
        assert tweet_storage.temp_dir.exists()
