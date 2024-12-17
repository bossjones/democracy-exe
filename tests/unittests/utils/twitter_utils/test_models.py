"""Tests for Twitter data models."""

from __future__ import annotations

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
from democracy_exe.utils.twitter_utils.types import TweetDownloadMode


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


class TestMediaType:
    """Tests for MediaType enum."""

    @pytest.mark.parametrize(
        "mime_type,expected",
        [
            ("image/jpeg", MediaType.IMAGE),
            ("image/png", MediaType.IMAGE),
            ("image/webp", MediaType.IMAGE),
            ("video/mp4", MediaType.VIDEO),
            ("video/webm", MediaType.VIDEO),
            ("image/gif", MediaType.GIF),
            ("audio/mp3", MediaType.AUDIO),
            ("audio/wav", MediaType.AUDIO),
            ("audio/ogg", MediaType.AUDIO),
        ],
    )
    def test_from_mime_type(self, mime_type: str, expected: MediaType) -> None:
        """Test MediaType.from_mime_type method.

        Args:
            mime_type: MIME type to test
            expected: Expected MediaType
        """
        assert MediaType.from_mime_type(mime_type) == expected

    @pytest.mark.parametrize(
        "mime_type",
        [
            "invalid/type",
            "text/plain",
            "application/json",
            "",
            "image",
            "video",
        ],
    )
    def test_from_mime_type_invalid(self, mime_type: str) -> None:
        """Test MediaType.from_mime_type with invalid MIME types.

        Args:
            mime_type: Invalid MIME type to test
        """
        with pytest.raises(ValueError, match=f"Unsupported MIME type: {mime_type}"):
            MediaType.from_mime_type(mime_type)


class TestMediaItem:
    """Tests for MediaItem class."""

    @pytest.fixture
    def media_item(self, tmp_path: Path) -> MediaItem:
        """Create a sample media item.

        Args:
            tmp_path: Temporary directory path

        Returns:
            Sample media item
        """
        local_path = tmp_path / "test.jpg"
        local_path.write_text("test")
        return MediaItem(
            url=TEST_MEDIA_URL,
            type=MediaType.IMAGE,
            local_path=local_path,
            size=1024,
            width=800,
            height=600,
            duration=None,
        )

    def test_is_downloaded(self, media_item: MediaItem, tmp_path: Path) -> None:
        """Test is_downloaded property.

        Args:
            media_item: Sample media item
            tmp_path: Temporary directory path
        """
        assert media_item.is_downloaded is True

        # Test with non-existent file
        media_item.local_path = tmp_path / "nonexistent.jpg"
        assert media_item.is_downloaded is False

        # Test with None path
        media_item.local_path = None
        assert media_item.is_downloaded is False

    def test_dimensions(self, media_item: MediaItem) -> None:
        """Test dimensions property.

        Args:
            media_item: Sample media item
        """
        # Test with both dimensions
        assert media_item.dimensions == (800, 600)

        # Test with missing width
        media_item.width = None
        assert media_item.dimensions is None

        # Test with missing height
        media_item.width = 800
        media_item.height = None
        assert media_item.dimensions is None

        # Test with both missing
        media_item.width = None
        assert media_item.dimensions is None

    @pytest.mark.parametrize("media_type", list(MediaType))
    def test_create_with_different_types(self, media_type: MediaType) -> None:
        """Test creating MediaItem with different types.

        Args:
            media_type: Media type to test
        """
        item = MediaItem(url=TEST_MEDIA_URL, type=media_type)
        assert item.type == media_type
        assert item.url == TEST_MEDIA_URL
        assert item.local_path is None
        assert item.size is None
        assert item.width is None
        assert item.height is None
        assert item.duration is None


class TestTweetCard:
    """Tests for TweetCard class."""

    @pytest.fixture
    def tweet_card(self, tmp_path: Path) -> TweetCard:
        """Create a sample tweet card.

        Args:
            tmp_path: Temporary directory path

        Returns:
            Sample tweet card
        """
        local_path = tmp_path / "card.jpg"
        local_path.write_text("test")
        return TweetCard(
            url=TEST_CARD_URL,
            title=TEST_CARD_TITLE,
            description=TEST_CARD_DESCRIPTION,
            image_url=TEST_CARD_IMAGE,
            local_image=local_path,
        )

    def test_has_image(self, tweet_card: TweetCard) -> None:
        """Test has_image property.

        Args:
            tweet_card: Sample tweet card
        """
        assert tweet_card.has_image is True

        # Test with None image URL
        tweet_card.image_url = None
        assert tweet_card.has_image is False

        # Test with empty image URL
        tweet_card.image_url = ""
        assert tweet_card.has_image is False

    def test_is_downloaded(self, tweet_card: TweetCard, tmp_path: Path) -> None:
        """Test is_downloaded property.

        Args:
            tweet_card: Sample tweet card
            tmp_path: Temporary directory path
        """
        assert tweet_card.is_downloaded is True

        # Test with non-existent file
        tweet_card.local_image = tmp_path / "nonexistent.jpg"
        assert tweet_card.is_downloaded is False

        # Test with None path
        tweet_card.local_image = None
        assert tweet_card.is_downloaded is False

    def test_create_minimal(self) -> None:
        """Test creating TweetCard with minimal data."""
        card = TweetCard(url=TEST_CARD_URL, title=TEST_CARD_TITLE)
        assert card.url == TEST_CARD_URL
        assert card.title == TEST_CARD_TITLE
        assert card.description is None
        assert card.image_url is None
        assert card.local_image is None
        assert card.has_image is False
        assert card.is_downloaded is False


class TestTweet:
    """Tests for Tweet class."""

    @pytest.fixture
    def tweet(self, tmp_path: Path) -> Tweet:
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

    def test_has_media(self, tweet: Tweet) -> None:
        """Test has_media property.

        Args:
            tweet: Sample tweet
        """
        assert tweet.has_media is True

        # Test with empty media list
        tweet.media = []
        assert tweet.has_media is False

        # Test with None media
        tweet.media = None  # type: ignore
        assert tweet.has_media is False

    def test_has_card(self, tweet: Tweet) -> None:
        """Test has_card property.

        Args:
            tweet: Sample tweet
        """
        assert tweet.has_card is True

        # Test with None card
        tweet.card = None
        assert tweet.has_card is False

    def test_is_quote(self, tweet: Tweet) -> None:
        """Test is_quote property.

        Args:
            tweet: Sample tweet
        """
        assert tweet.is_quote is False

        # Test with quoted tweet
        tweet.quoted_tweet = Tweet(
            id="quoted",
            author=TEST_AUTHOR,
            content="Quoted tweet",
            created_at=TEST_CREATED_AT,
            url=TEST_URL,
        )
        assert tweet.is_quote is True

        # Test with None quoted tweet
        tweet.quoted_tweet = None
        assert tweet.is_quote is False

    def test_to_dict(self, tweet: Tweet) -> None:
        """Test to_dict method.

        Args:
            tweet: Sample tweet
        """
        result = tweet.to_dict()

        # Test basic fields
        assert isinstance(result, dict)
        assert result["id"] == tweet.id
        assert result["author"] == tweet.author
        assert result["content"] == tweet.content
        assert result["created_at"] == tweet.created_at.isoformat()
        assert result["url"] == tweet.url
        assert result["retweet_count"] == tweet.retweet_count
        assert result["like_count"] == tweet.like_count
        assert result["reply_count"] == tweet.reply_count

        # Test media URLs
        assert result["media_urls"] == [m.url for m in tweet.media]

        # Test card fields
        assert result["card_url"] == tweet.card.url
        assert result["card_description"] == tweet.card.description
        assert result["card_image"] == tweet.card.image_url

    def test_to_dict_minimal(self) -> None:
        """Test to_dict method with minimal tweet."""
        tweet = Tweet(
            id=TEST_TWEET_ID,
            author=TEST_AUTHOR,
            content=TEST_CONTENT,
            created_at=TEST_CREATED_AT,
            url=TEST_URL,
        )
        result = tweet.to_dict()

        # Check only required fields are present
        assert set(result.keys()) == {
            "id",
            "author",
            "content",
            "created_at",
            "url",
            "retweet_count",
            "like_count",
            "reply_count",
        }


class TestTweetThread:
    """Tests for TweetThread class."""

    @pytest.fixture
    def tweet_thread(self) -> TweetThread:
        """Create a sample tweet thread.

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

    def test_length(self, tweet_thread: TweetThread) -> None:
        """Test length property.

        Args:
            tweet_thread: Sample tweet thread
        """
        assert tweet_thread.length == 3

        # Test with empty thread
        tweet_thread.tweets = []
        assert tweet_thread.length == 0

        # Test with None tweets
        tweet_thread.tweets = None  # type: ignore
        assert tweet_thread.length == 0

    def test_first_tweet(self, tweet_thread: TweetThread) -> None:
        """Test first_tweet property.

        Args:
            tweet_thread: Sample tweet thread
        """
        assert tweet_thread.first_tweet is not None
        assert tweet_thread.first_tweet.id == f"{TEST_TWEET_ID}0"

        # Test with empty thread
        tweet_thread.tweets = []
        assert tweet_thread.first_tweet is None

    def test_last_tweet(self, tweet_thread: TweetThread) -> None:
        """Test last_tweet property.

        Args:
            tweet_thread: Sample tweet thread
        """
        assert tweet_thread.last_tweet is not None
        assert tweet_thread.last_tweet.id == f"{TEST_TWEET_ID}2"

        # Test with empty thread
        tweet_thread.tweets = []
        assert tweet_thread.last_tweet is None

    def test_to_dict(self, tweet_thread: TweetThread) -> None:
        """Test to_dict method.

        Args:
            tweet_thread: Sample tweet thread
        """
        result = tweet_thread.to_dict()

        # Test basic fields
        assert isinstance(result, dict)
        assert result["author"] == tweet_thread.author
        assert result["created_at"] == tweet_thread.created_at.isoformat()

        # Test tweets list
        assert len(result["tweets"]) == len(tweet_thread.tweets)
        assert all(isinstance(t, dict) for t in result["tweets"])
        assert all(t["id"].startswith(TEST_TWEET_ID) for t in result["tweets"])

    def test_to_dict_empty(self) -> None:
        """Test to_dict method with empty thread."""
        thread = TweetThread(tweets=[], author=TEST_AUTHOR, created_at=TEST_CREATED_AT)
        result = thread.to_dict()

        assert result["tweets"] == []
        assert result["author"] == TEST_AUTHOR
        assert result["created_at"] == TEST_CREATED_AT.isoformat()


class TestDownloadedContent:
    """Tests for DownloadedContent class."""

    @pytest.fixture
    def tweet_thread(self) -> TweetThread:
        """Create a sample tweet thread.

        Returns:
            Sample tweet thread
        """
        tweets = [
            Tweet(
                id=f"123456789{i}",
                author="test_user",
                content=f"Tweet {i + 1}",
                created_at=datetime(2024, 1, 1, 12, 0, 0),
                url="https://twitter.com/test_user/status/123456789",
            )
            for i in range(3)
        ]
        return TweetThread(tweets=tweets, author="test_user", created_at=datetime(2024, 1, 1, 12, 0, 0))

    @pytest.fixture
    def downloaded_content(self, tweet_thread: TweetThread, tmp_path: Path) -> DownloadedContent:
        """Create a sample downloaded content.

        Args:
            tweet_thread: Sample tweet thread
            tmp_path: Temporary directory path

        Returns:
            Sample downloaded content
        """
        return DownloadedContent(
            mode="thread",
            content=tweet_thread,
            local_files=[
                tmp_path / "file1.jpg",
                tmp_path / "file2.jpg",
            ],
        )

    def test_success(self, downloaded_content: DownloadedContent) -> None:
        """Test success property.

        Args:
            downloaded_content: Sample downloaded content
        """
        # Test successful state
        assert downloaded_content.success is True

        # Test with missing content
        downloaded_content.content = None
        assert downloaded_content.success is False

        # Test with error
        downloaded_content.error = "Error"
        assert downloaded_content.success is False

        # Test with both content and error
        downloaded_content.content = Tweet(
            id=TEST_TWEET_ID,
            author=TEST_AUTHOR,
            content=TEST_CONTENT,
            created_at=TEST_CREATED_AT,
            url=TEST_URL,
        )
        assert downloaded_content.success is False

    def test_has_files(self, downloaded_content: DownloadedContent) -> None:
        """Test has_files property.

        Args:
            downloaded_content: Sample downloaded content
        """
        assert downloaded_content.has_files is True

        # Test with empty files list
        downloaded_content.local_files = []
        assert downloaded_content.has_files is False

        # Test with None files
        downloaded_content.local_files = None  # type: ignore
        assert downloaded_content.has_files is False

    def test_to_dict(self, downloaded_content: DownloadedContent) -> None:
        """Test to_dict method.

        Args:
            downloaded_content: Sample downloaded content
        """
        result = downloaded_content.to_dict()

        # Test basic fields
        assert isinstance(result, dict)
        assert result["success"] == downloaded_content.success
        assert result["mode"] == downloaded_content.mode
        assert len(result["local_files"]) == len(downloaded_content.local_files)
        assert all(isinstance(f, str) for f in result["local_files"])

        # Test metadata
        assert "metadata" in result
        assert isinstance(result["metadata"], dict)

    def test_to_dict_with_error(self) -> None:
        """Test to_dict method with error state."""
        content = DownloadedContent(mode="single", error="Test error")
        result = content.to_dict()

        assert result["success"] is False
        assert result["error"] == "Test error"
        assert "metadata" not in result

    @pytest.mark.parametrize("mode", ["single", "thread", "card"])
    def test_create_with_different_modes(self, mode: TweetDownloadMode) -> None:
        """Test creating DownloadedContent with different modes.

        Args:
            mode: Download mode to test
        """
        content = DownloadedContent(mode=mode)
        assert content.mode == mode
        assert content.content is None
        assert content.local_files == []
        assert content.error is None
        assert content.success is False
        assert content.has_files is False
