"""Tests for Twitter client utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from typing import TYPE_CHECKING, Any, Dict

from loguru import logger

import pytest

from democracy_exe.utils.twitter_utils.client import GalleryDLError, NoExtractorError, NoTweetDataError, TwitterClient
from democracy_exe.utils.twitter_utils.types import TweetMetadata


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_tweet_item() -> dict[str, Any]:
    """Create mock tweet data.

    Returns:
        Mock tweet item dictionary
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
        TwitterClient instance
    """
    return TwitterClient(auth_token="test_token")  # noqa: S106


class TestTwitterClient:
    """Test suite for Twitter client."""

    def test_extract_tweet_id(self, twitter_client: TwitterClient) -> None:
        """Test tweet ID extraction from URLs.

        Args:
            twitter_client: Twitter client fixture
        """
        urls = [
            "https://twitter.com/user/status/123456789",
            "https://x.com/user/status/123456789",
            "https://twitter.com/user/status/123456789?s=20",
            "invalid_url",
        ]

        assert twitter_client.extract_tweet_id(urls[0]) == "123456789"
        assert twitter_client.extract_tweet_id(urls[1]) == "123456789"
        assert twitter_client.extract_tweet_id(urls[2]) == "123456789"
        assert twitter_client.extract_tweet_id(urls[3]) is None

    def test_parse_tweet_item(self, twitter_client: TwitterClient, mock_tweet_item: dict[str, Any]) -> None:
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

    def test_parse_tweet_item_missing_field(self, twitter_client: TwitterClient) -> None:
        """Test parsing tweet item with missing required field.

        Args:
            twitter_client: Twitter client fixture
        """
        invalid_item = {
            "tweet_id": 123456789,
            # Missing required fields
        }

        with pytest.raises(ValueError, match="Missing required field"):
            twitter_client._parse_tweet_item(invalid_item)  # type: ignore

    @pytest.mark.vcr()
    def test_get_tweet_metadata(
        self, twitter_client: TwitterClient, mocker: MockerFixture, mock_tweet_item: dict[str, Any]
    ) -> None:
        """Test tweet metadata fetching.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
            mock_tweet_item: Mock tweet data fixture
        """
        mock_extractor = mocker.MagicMock()
        mock_extractor.__iter__.return_value = iter([mock_tweet_item])
        mocker.patch("gallery_dl.extractor.find", return_value=mock_extractor)

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
        mocker.patch("gallery_dl.extractor.find", return_value=None)

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
        mocker.patch("gallery_dl.extractor.find", return_value=mock_extractor)

        with pytest.raises(NoTweetDataError, match="No tweet data found"):
            twitter_client.get_tweet_metadata("https://twitter.com/user/status/123")

    @pytest.mark.vcr()
    def test_get_thread_tweets(
        self, twitter_client: TwitterClient, mocker: MockerFixture, mock_tweet_item: dict[str, Any]
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
        mocker.patch("gallery_dl.extractor.find", return_value=mock_extractor)

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
        mocker.patch("gallery_dl.extractor.find", return_value=None)

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
        mocker.patch("gallery_dl.extractor.find", return_value=mock_extractor)

        with pytest.raises(NoTweetDataError, match="No tweets found in thread"):
            twitter_client.get_thread_tweets("https://twitter.com/user/status/123")

    def test_validate_tweet(
        self, twitter_client: TwitterClient, mocker: MockerFixture, mock_tweet_item: dict[str, Any]
    ) -> None:
        """Test tweet validation.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
            mock_tweet_item: Mock tweet data fixture
        """
        mock_extractor = mocker.MagicMock()
        mock_extractor.__iter__.return_value = iter([mock_tweet_item])
        mocker.patch("gallery_dl.extractor.find", return_value=mock_extractor)

        assert twitter_client.validate_tweet("https://twitter.com/user/status/123456789")

        # Test invalid URL
        mocker.patch("gallery_dl.extractor.find", return_value=None)
        assert not twitter_client.validate_tweet("invalid_url")
