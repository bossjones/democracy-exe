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
    TweetStorage,
    get_download_path,
    get_media_path,
    get_metadata_path,
    get_storage_path,
    get_tweet_path,
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
def tweet_storage(storage_root: Path) -> TweetStorage:
    """Create a TweetStorage instance.

    Args:
        storage_root: Storage root directory

    Returns:
        TweetStorage instance
    """
    return TweetStorage(storage_root)


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


class TestPathUtilities:
    """Tests for path utility functions."""

    def test_get_storage_path(self, storage_root: Path) -> None:
        """Test get_storage_path function.

        Args:
            storage_root: Storage root directory
        """
        path = get_storage_path(storage_root, TEST_AUTHOR)
        assert path == storage_root / TEST_AUTHOR
        assert path.is_dir()

    def test_get_storage_path_special_chars(self, storage_root: Path) -> None:
        """Test get_storage_path with special characters.

        Args:
            storage_root: Storage root directory
        """
        special_author = "test@user/with\\special:chars"
        path = get_storage_path(storage_root, special_author)
        assert path == storage_root / special_author
        assert path.is_dir()

    def test_get_tweet_path(self, storage_root: Path) -> None:
        """Test get_tweet_path function.

        Args:
            storage_root: Storage root directory
        """
        path = get_tweet_path(storage_root, TEST_AUTHOR, TEST_TWEET_ID)
        assert path == storage_root / TEST_AUTHOR / TEST_TWEET_ID
        assert path.is_dir()

    def test_get_tweet_path_special_chars(self, storage_root: Path) -> None:
        """Test get_tweet_path with special characters.

        Args:
            storage_root: Storage root directory
        """
        special_id = "123/456\\789:test"
        path = get_tweet_path(storage_root, TEST_AUTHOR, special_id)
        assert path == storage_root / TEST_AUTHOR / special_id
        assert path.is_dir()

    def test_get_media_path(self, storage_root: Path) -> None:
        """Test get_media_path function.

        Args:
            storage_root: Storage root directory
        """
        path = get_media_path(storage_root, TEST_AUTHOR, TEST_TWEET_ID)
        assert path == storage_root / TEST_AUTHOR / TEST_TWEET_ID / "media"
        assert path.is_dir()

    def test_get_metadata_path(self, storage_root: Path) -> None:
        """Test get_metadata_path function.

        Args:
            storage_root: Storage root directory
        """
        path = get_metadata_path(storage_root, TEST_AUTHOR, TEST_TWEET_ID)
        assert path == storage_root / TEST_AUTHOR / TEST_TWEET_ID / "metadata.json"

    def test_get_download_path(self, storage_root: Path) -> None:
        """Test get_download_path function.

        Args:
            storage_root: Storage root directory
        """
        path = get_download_path(storage_root, TEST_AUTHOR, TEST_TWEET_ID)
        assert path == storage_root / TEST_AUTHOR / TEST_TWEET_ID / "download.json"


class TestTweetStorage:
    """Tests for TweetStorage class."""

    def test_init(self, storage_root: Path) -> None:
        """Test TweetStorage initialization.

        Args:
            storage_root: Storage root directory
        """
        storage = TweetStorage(storage_root)
        assert storage.root == storage_root
        assert storage.root.exists()
        assert storage.root.is_dir()

    def test_init_nonexistent_root(self, tmp_path: Path) -> None:
        """Test TweetStorage initialization with nonexistent root.

        Args:
            tmp_path: Temporary directory path
        """
        storage_dir = tmp_path / "nonexistent"
        storage = TweetStorage(storage_dir)
        assert storage.root.exists()
        assert storage.root.is_dir()

    def test_init_with_file(self, tmp_path: Path) -> None:
        """Test TweetStorage initialization with file path.

        Args:
            tmp_path: Temporary directory path
        """
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        with pytest.raises(NotADirectoryError):
            TweetStorage(file_path)

    def test_get_tweet_dir(self, tweet_storage: TweetStorage) -> None:
        """Test get_tweet_dir method.

        Args:
            tweet_storage: TweetStorage instance
        """
        path = tweet_storage.get_tweet_dir(TEST_AUTHOR, TEST_TWEET_ID)
        assert path == tweet_storage.root / TEST_AUTHOR / TEST_TWEET_ID
        assert path.exists()
        assert path.is_dir()

    def test_get_media_dir(self, tweet_storage: TweetStorage) -> None:
        """Test get_media_dir method.

        Args:
            tweet_storage: TweetStorage instance
        """
        path = tweet_storage.get_media_dir(TEST_AUTHOR, TEST_TWEET_ID)
        assert path == tweet_storage.root / TEST_AUTHOR / TEST_TWEET_ID / "media"
        assert path.exists()
        assert path.is_dir()

    def test_save_metadata(self, tweet_storage: TweetStorage, sample_tweet: Tweet) -> None:
        """Test save_metadata method.

        Args:
            tweet_storage: TweetStorage instance
            sample_tweet: Sample tweet
        """
        metadata = sample_tweet.to_dict()
        tweet_storage.save_metadata(TEST_AUTHOR, TEST_TWEET_ID, metadata)

        metadata_path = tweet_storage.get_metadata_path(TEST_AUTHOR, TEST_TWEET_ID)
        assert metadata_path.exists()

        # Verify saved data
        saved_data = json.loads(metadata_path.read_text())
        assert saved_data == metadata

    def test_save_metadata_invalid_json(self, tweet_storage: TweetStorage) -> None:
        """Test save_metadata with invalid JSON data.

        Args:
            tweet_storage: TweetStorage instance
        """

        class NonSerializable:
            pass

        with pytest.raises(TypeError):
            tweet_storage.save_metadata(TEST_AUTHOR, TEST_TWEET_ID, {"invalid": NonSerializable()})

    def test_save_download_info(self, tweet_storage: TweetStorage, sample_tweet: Tweet) -> None:
        """Test save_download_info method.

        Args:
            tweet_storage: TweetStorage instance
            sample_tweet: Sample tweet
        """
        download_info = DownloadedContent(
            mode="single",
            content=sample_tweet,
            local_files=[Path("test.jpg")],
        ).to_dict()
        tweet_storage.save_download_info(TEST_AUTHOR, TEST_TWEET_ID, download_info)

        download_path = tweet_storage.get_download_path(TEST_AUTHOR, TEST_TWEET_ID)
        assert download_path.exists()

        # Verify saved data
        saved_data = json.loads(download_path.read_text())
        assert saved_data == download_info

    def test_save_download_info_invalid_json(self, tweet_storage: TweetStorage) -> None:
        """Test save_download_info with invalid JSON data.

        Args:
            tweet_storage: TweetStorage instance
        """

        class NonSerializable:
            pass

        with pytest.raises(TypeError):
            tweet_storage.save_download_info(TEST_AUTHOR, TEST_TWEET_ID, {"invalid": NonSerializable()})

    def test_load_metadata(self, tweet_storage: TweetStorage, sample_tweet: Tweet) -> None:
        """Test load_metadata method.

        Args:
            tweet_storage: TweetStorage instance
            sample_tweet: Sample tweet
        """
        # Save metadata first
        metadata = sample_tweet.to_dict()
        tweet_storage.save_metadata(TEST_AUTHOR, TEST_TWEET_ID, metadata)

        # Load and verify
        loaded_data = tweet_storage.load_metadata(TEST_AUTHOR, TEST_TWEET_ID)
        assert loaded_data == metadata

    def test_load_metadata_nonexistent(self, tweet_storage: TweetStorage) -> None:
        """Test load_metadata with nonexistent file.

        Args:
            tweet_storage: TweetStorage instance
        """
        with pytest.raises(FileNotFoundError):
            tweet_storage.load_metadata(TEST_AUTHOR, "nonexistent")

    def test_load_metadata_invalid_json(self, tweet_storage: TweetStorage) -> None:
        """Test load_metadata with invalid JSON file.

        Args:
            tweet_storage: TweetStorage instance
        """
        # Create invalid JSON file
        metadata_path = tweet_storage.get_metadata_path(TEST_AUTHOR, TEST_TWEET_ID)
        metadata_path.parent.mkdir(parents=True)
        metadata_path.write_text("invalid json")

        with pytest.raises(json.JSONDecodeError):
            tweet_storage.load_metadata(TEST_AUTHOR, TEST_TWEET_ID)

    def test_load_download_info(self, tweet_storage: TweetStorage, sample_tweet: Tweet) -> None:
        """Test load_download_info method.

        Args:
            tweet_storage: TweetStorage instance
            sample_tweet: Sample tweet
        """
        # Save download info first
        download_info = DownloadedContent(
            mode="single",
            content=sample_tweet,
            local_files=[Path("test.jpg")],
        ).to_dict()
        tweet_storage.save_download_info(TEST_AUTHOR, TEST_TWEET_ID, download_info)

        # Load and verify
        loaded_data = tweet_storage.load_download_info(TEST_AUTHOR, TEST_TWEET_ID)
        assert loaded_data == download_info

    def test_load_download_info_nonexistent(self, tweet_storage: TweetStorage) -> None:
        """Test load_download_info with nonexistent file.

        Args:
            tweet_storage: TweetStorage instance
        """
        with pytest.raises(FileNotFoundError):
            tweet_storage.load_download_info(TEST_AUTHOR, "nonexistent")

    def test_load_download_info_invalid_json(self, tweet_storage: TweetStorage) -> None:
        """Test load_download_info with invalid JSON file.

        Args:
            tweet_storage: TweetStorage instance
        """
        # Create invalid JSON file
        download_path = tweet_storage.get_download_path(TEST_AUTHOR, TEST_TWEET_ID)
        download_path.parent.mkdir(parents=True)
        download_path.write_text("invalid json")

        with pytest.raises(json.JSONDecodeError):
            tweet_storage.load_download_info(TEST_AUTHOR, TEST_TWEET_ID)

    def test_get_media_file_path(self, tweet_storage: TweetStorage) -> None:
        """Test get_media_file_path method.

        Args:
            tweet_storage: TweetStorage instance
        """
        filename = "test.jpg"
        path = tweet_storage.get_media_file_path(TEST_AUTHOR, TEST_TWEET_ID, filename)
        assert path == tweet_storage.root / TEST_AUTHOR / TEST_TWEET_ID / "media" / filename

    def test_get_media_file_path_special_chars(self, tweet_storage: TweetStorage) -> None:
        """Test get_media_file_path with special characters.

        Args:
            tweet_storage: TweetStorage instance
        """
        filename = "test/with\\special:chars.jpg"
        path = tweet_storage.get_media_file_path(TEST_AUTHOR, TEST_TWEET_ID, filename)
        assert path == tweet_storage.root / TEST_AUTHOR / TEST_TWEET_ID / "media" / filename

    def test_cleanup_empty_dirs(self, tweet_storage: TweetStorage) -> None:
        """Test cleanup_empty_dirs method.

        Args:
            tweet_storage: TweetStorage instance
        """
        # Create some empty directories
        empty_dir = tweet_storage.root / "empty_user" / "empty_tweet"
        empty_dir.mkdir(parents=True)

        # Create a directory with content
        non_empty_dir = tweet_storage.root / "user" / "tweet"
        non_empty_dir.mkdir(parents=True)
        (non_empty_dir / "test.txt").write_text("test")

        tweet_storage.cleanup_empty_dirs()

        # Empty directory should be removed
        assert not empty_dir.exists()
        assert not empty_dir.parent.exists()

        # Non-empty directory should remain
        assert non_empty_dir.exists()
        assert (non_empty_dir / "test.txt").exists()

    def test_cleanup_empty_dirs_with_permission_error(self, tweet_storage: TweetStorage, mocker: MockerFixture) -> None:
        """Test cleanup_empty_dirs with permission error.

        Args:
            tweet_storage: TweetStorage instance
            mocker: Pytest mocker fixture
        """
        # Mock rmdir to raise PermissionError
        mocker.patch.object(Path, "rmdir", side_effect=PermissionError)

        # Create empty directory
        empty_dir = tweet_storage.root / "empty_user" / "empty_tweet"
        empty_dir.mkdir(parents=True)

        # Should not raise exception
        tweet_storage.cleanup_empty_dirs()

        # Directory should still exist
        assert empty_dir.exists()

    def test_get_all_tweets(self, tweet_storage: TweetStorage, sample_tweet: Tweet) -> None:
        """Test get_all_tweets method.

        Args:
            tweet_storage: TweetStorage instance
            sample_tweet: Sample tweet
        """
        # Save some tweets
        for i in range(3):
            tweet_id = f"{TEST_TWEET_ID}{i}"
            metadata = sample_tweet.to_dict()
            metadata["id"] = tweet_id
            tweet_storage.save_metadata(TEST_AUTHOR, tweet_id, metadata)

        # Get all tweets
        tweets = tweet_storage.get_all_tweets(TEST_AUTHOR)
        assert len(tweets) == 3
        assert all(isinstance(t, dict) for t in tweets)
        assert all(t["author"] == TEST_AUTHOR for t in tweets)

    def test_get_all_tweets_with_invalid_files(self, tweet_storage: TweetStorage, sample_tweet: Tweet) -> None:
        """Test get_all_tweets with some invalid files.

        Args:
            tweet_storage: TweetStorage instance
            sample_tweet: Sample tweet
        """
        # Save valid tweet
        metadata = sample_tweet.to_dict()
        tweet_storage.save_metadata(TEST_AUTHOR, TEST_TWEET_ID, metadata)

        # Create invalid JSON file
        invalid_path = tweet_storage.get_metadata_path(TEST_AUTHOR, "invalid")
        invalid_path.parent.mkdir(parents=True)
        invalid_path.write_text("invalid json")

        # Should skip invalid file and return valid tweets
        tweets = tweet_storage.get_all_tweets(TEST_AUTHOR)
        assert len(tweets) == 1
        assert tweets[0] == metadata

    def test_get_all_tweets_empty(self, tweet_storage: TweetStorage) -> None:
        """Test get_all_tweets with no tweets.

        Args:
            tweet_storage: TweetStorage instance
        """
        tweets = tweet_storage.get_all_tweets(TEST_AUTHOR)
        assert tweets == []

    def test_get_all_downloads(self, tweet_storage: TweetStorage, sample_tweet: Tweet) -> None:
        """Test get_all_downloads method.

        Args:
            tweet_storage: TweetStorage instance
            sample_tweet: Sample tweet
        """
        # Save some downloads
        for i in range(3):
            tweet_id = f"{TEST_TWEET_ID}{i}"
            download_info = DownloadedContent(
                mode="single",
                content=sample_tweet,
                local_files=[Path("test.jpg")],
            ).to_dict()
            tweet_storage.save_download_info(TEST_AUTHOR, tweet_id, download_info)

        # Get all downloads
        downloads = tweet_storage.get_all_downloads(TEST_AUTHOR)
        assert len(downloads) == 3
        assert all(isinstance(d, dict) for d in downloads)
        assert all(d["success"] is True for d in downloads)

    def test_get_all_downloads_with_invalid_files(self, tweet_storage: TweetStorage, sample_tweet: Tweet) -> None:
        """Test get_all_downloads with some invalid files.

        Args:
            tweet_storage: TweetStorage instance
            sample_tweet: Sample tweet
        """
        # Save valid download
        download_info = DownloadedContent(
            mode="single",
            content=sample_tweet,
            local_files=[Path("test.jpg")],
        ).to_dict()
        tweet_storage.save_download_info(TEST_AUTHOR, TEST_TWEET_ID, download_info)

        # Create invalid JSON file
        invalid_path = tweet_storage.get_download_path(TEST_AUTHOR, "invalid")
        invalid_path.parent.mkdir(parents=True)
        invalid_path.write_text("invalid json")

        # Should skip invalid file and return valid downloads
        downloads = tweet_storage.get_all_downloads(TEST_AUTHOR)
        assert len(downloads) == 1
        assert downloads[0] == download_info

    def test_get_all_downloads_empty(self, tweet_storage: TweetStorage) -> None:
        """Test get_all_downloads with no downloads.

        Args:
            tweet_storage: TweetStorage instance
        """
        downloads = tweet_storage.get_all_downloads(TEST_AUTHOR)
        assert downloads == []
