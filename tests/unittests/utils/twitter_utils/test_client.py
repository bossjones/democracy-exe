"""Tests for Twitter client utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Dict, List

# from loguru import logger
import structlog


logger = structlog.get_logger(__name__)

import pytest

from democracy_exe.utils.twitter_utils.client import (
    GalleryDLError,
    GalleryDLTweetItem,
    NoExtractorError,
    NoTweetDataError,
    TwitterClient,
)
from democracy_exe.utils.twitter_utils.types import TweetMetadata


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_tweet_item() -> GalleryDLTweetItem:
    """Create mock tweet data.

    Returns:
        Mock tweet item dictionary with required fields
    """
    return {
        "tweet_id": 123456789,
        "tweet_url": "https://twitter.com/user/status/123456789",
        "content": "Test tweet content",
        "date": datetime(2024, 3, 10, 12, 0, 0, tzinfo=UTC),
        "author": {"name": "Test User"},
        "media": [{"url": "https://example.com/image.jpg"}],
    }


@pytest.fixture
def twitter_client() -> TwitterClient:
    """Create test Twitter client.

    Returns:
        TwitterClient instance with test auth token
    """
    return TwitterClient(auth_token="test_token")  # noqa: S106


class TestTwitterClient:
    """Test suite for Twitter client."""

    def test_extract_tweet_id(self, twitter_client: TwitterClient) -> None:
        """Test tweet ID extraction from various URL formats.

        Args:
            twitter_client: Twitter client fixture
        """
        test_cases = [
            ("https://twitter.com/user/status/123456789", "123456789"),
            ("https://x.com/user/status/123456789", "123456789"),
            ("https://twitter.com/user/status/123456789?s=20", "123456789"),
            ("https://x.com/user/status/123456789?s=20&t=abc", "123456789"),
            ("invalid_url", None),
            ("https://twitter.com/user/", None),
            ("https://x.com/user/status/", None),
        ]

        for url, expected in test_cases:
            assert twitter_client.extract_tweet_id(url) == expected

    def test_parse_tweet_item(self, twitter_client: TwitterClient, mock_tweet_item: GalleryDLTweetItem) -> None:
        """Test parsing of gallery-dl tweet items.

        Args:
            twitter_client: Twitter client fixture
            mock_tweet_item: Mock tweet data fixture
        """
        metadata = twitter_client._parse_tweet_item(mock_tweet_item)

        assert metadata["id"] == "123456789"
        assert metadata["url"] == "https://twitter.com/user/status/123456789"
        assert metadata["author"] == "Test User"
        assert metadata["content"] == "Test tweet content"
        assert len(metadata["media_urls"]) == 1
        assert metadata["media_urls"][0] == "https://example.com/image.jpg"
        assert metadata["created_at"] == "2024-03-10T12:00:00+00:00"

    def test_parse_tweet_item_no_media(
        self, twitter_client: TwitterClient, mock_tweet_item: GalleryDLTweetItem
    ) -> None:
        """Test parsing tweet item without media.

        Args:
            twitter_client: Twitter client fixture
            mock_tweet_item: Mock tweet data fixture
        """
        mock_tweet_item["media"] = None
        metadata = twitter_client._parse_tweet_item(mock_tweet_item)
        assert metadata["media_urls"] == []

    def test_parse_tweet_item_missing_field(self, twitter_client: TwitterClient) -> None:
        """Test parsing tweet item with missing required fields.

        Args:
            twitter_client: Twitter client fixture
        """
        invalid_items = [
            {
                "tweet_url": "url",
                "content": "content",
                "date": datetime.now(UTC),
                "author": {"name": "author"},
            },  # Missing tweet_id
            {
                "tweet_id": 123,
                "content": "content",
                "date": datetime.now(UTC),
                "author": {"name": "author"},
            },  # Missing tweet_url
            {
                "tweet_id": 123,
                "tweet_url": "url",
                "date": datetime.now(UTC),
                "author": {"name": "author"},
            },  # Missing content
            {"tweet_id": 123, "tweet_url": "url", "content": "content", "author": {"name": "author"}},  # Missing date
            {"tweet_id": 123, "tweet_url": "url", "content": "content", "date": datetime.now(UTC)},  # Missing author
        ]

        for item in invalid_items:
            with pytest.raises(ValueError, match="Missing required field"):
                twitter_client._parse_tweet_item(item)  # type: ignore

    @pytest.mark.vcr()
    def test_get_tweet_metadata(
        self, twitter_client: TwitterClient, mocker: MockerFixture, mock_tweet_item: GalleryDLTweetItem
    ) -> None:
        """Test tweet metadata fetching.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
            mock_tweet_item: Mock tweet data fixture
        """
        mock_extractor = mocker.MagicMock()
        mock_extractor.__iter__.return_value = iter([mock_tweet_item])
        mocker.patch("gallery_dl.extractor.twitter.TwitterExtractor", return_value=mock_extractor)

        metadata = twitter_client.get_tweet_metadata("https://twitter.com/user/status/123456789")

        assert metadata["id"] == "123456789"
        assert metadata["author"] == "Test User"
        assert metadata["content"] == "Test tweet content"
        assert len(metadata["media_urls"]) == 1
        assert metadata["created_at"] == "2024-03-10T12:00:00+00:00"

    @pytest.mark.vcr()
    def test_get_tweet_metadata_no_extractor(self, twitter_client: TwitterClient, mocker: MockerFixture) -> None:
        """Test tweet metadata fetching with no extractor.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
        """

        def raise_no_extractor(*args: Any, **kwargs: Any) -> None:
            raise NoExtractorError("No suitable extractor found")

        mocker.patch(
            "gallery_dl.extractor.twitter.TwitterExtractor",
            side_effect=raise_no_extractor,
        )

        with pytest.raises(NoExtractorError, match="No suitable extractor found"):
            twitter_client.get_tweet_metadata("https://example.com")

    @pytest.mark.vcr()
    def test_get_tweet_metadata_no_data(self, twitter_client: TwitterClient, mocker: MockerFixture) -> None:
        """Test tweet metadata fetching with no data.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
        """
        mock_extractor = mocker.MagicMock()
        mock_extractor.__iter__.return_value = iter([])
        mocker.patch("gallery_dl.extractor.twitter.TwitterExtractor", return_value=mock_extractor)

        with pytest.raises(NoTweetDataError, match="No tweet data found"):
            twitter_client.get_tweet_metadata("https://twitter.com/user/status/123")

    @pytest.mark.vcr()
    def test_get_thread_tweets(
        self, twitter_client: TwitterClient, mocker: MockerFixture, mock_tweet_item: GalleryDLTweetItem
    ) -> None:
        """Test thread tweets fetching.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
            mock_tweet_item: Mock tweet data fixture
        """
        thread_items = [
            mock_tweet_item,
            {
                "tweet_id": 987654321,
                "tweet_url": "https://twitter.com/user/status/987654321",
                "content": "Reply tweet",
                "date": datetime(2024, 3, 10, 12, 1, 0, tzinfo=UTC),
                "author": {"name": "Test User"},
                "media": [],
            },
        ]

        mock_extractor = mocker.MagicMock()
        mock_extractor.__iter__.return_value = iter(thread_items)
        mocker.patch("gallery_dl.extractor.twitter.TwitterExtractor", return_value=mock_extractor)

        thread = twitter_client.get_thread_tweets("https://twitter.com/user/status/123456789")

        assert len(thread) == 2
        assert thread[0]["id"] == "123456789"
        assert thread[1]["id"] == "987654321"
        assert thread[0]["created_at"] == "2024-03-10T12:00:00+00:00"
        assert thread[1]["created_at"] == "2024-03-10T12:01:00+00:00"

    @pytest.mark.vcr()
    def test_get_thread_tweets_no_extractor(self, twitter_client: TwitterClient, mocker: MockerFixture) -> None:
        """Test thread tweets fetching with no extractor.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
        """

        def raise_no_extractor(*args: Any, **kwargs: Any) -> None:
            raise NoExtractorError("No suitable extractor found")

        mocker.patch(
            "gallery_dl.extractor.twitter.TwitterExtractor",
            side_effect=raise_no_extractor,
        )

        with pytest.raises(NoExtractorError, match="No suitable extractor found"):
            twitter_client.get_thread_tweets("https://example.com")

    @pytest.mark.vcr()
    def test_get_thread_tweets_no_data(self, twitter_client: TwitterClient, mocker: MockerFixture) -> None:
        """Test thread tweets fetching with no data.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
        """
        mock_extractor = mocker.MagicMock()
        mock_extractor.__iter__.return_value = iter([])
        mocker.patch("gallery_dl.extractor.twitter.TwitterExtractor", return_value=mock_extractor)

        with pytest.raises(NoTweetDataError, match="No tweets found in thread"):
            twitter_client.get_thread_tweets("https://twitter.com/user/status/123")

    def test_validate_tweet(
        self, twitter_client: TwitterClient, mocker: MockerFixture, mock_tweet_item: GalleryDLTweetItem
    ) -> None:
        """Test tweet validation.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
            mock_tweet_item: Mock tweet data fixture
        """
        mock_extractor = mocker.MagicMock()
        mock_extractor.__iter__.return_value = iter([mock_tweet_item])
        mocker.patch("gallery_dl.extractor.twitter.TwitterExtractor", return_value=mock_extractor)

        assert twitter_client.validate_tweet("https://twitter.com/user/status/123456789") is True

        # Test invalid URL
        def raise_no_extractor(*args: Any, **kwargs: Any) -> None:
            raise NoExtractorError("No suitable extractor found")

        mocker.patch(
            "gallery_dl.extractor.twitter.TwitterExtractor",
            side_effect=raise_no_extractor,
        )
        assert twitter_client.validate_tweet("invalid_url") is False

        # Test no data
        mock_extractor = mocker.MagicMock()
        mock_extractor.__iter__.return_value = iter([])
        mocker.patch("gallery_dl.extractor.twitter.TwitterExtractor", return_value=mock_extractor)
        assert twitter_client.validate_tweet("https://twitter.com/user/status/123456789") is False
