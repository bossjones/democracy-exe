"""Tests for Twitter client utilities."""

from __future__ import annotations

import json

from typing import TYPE_CHECKING

import aiohttp

from aioresponses import aioresponses

import pytest

from democracy_exe.utils.twitter_utils.client import TwitterClient


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def twitter_client() -> TwitterClient:
    """Create test Twitter client.

    Returns:
        TwitterClient instance
    """
    return TwitterClient(api_key="test_key", api_secret="test_secret", bearer_token="test_token")  # noqa: S106


@pytest.mark.asyncio
class TestTwitterClient:
    """Test suite for Twitter client."""

    async def test_extract_tweet_id(self, twitter_client: TwitterClient) -> None:
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

    async def test_validate_tweet(self, twitter_client: TwitterClient, mocker: MockerFixture) -> None:
        """Test tweet validation.

        Args:
            twitter_client: Twitter client fixture
            mocker: Pytest mocker fixture
        """
        mock_metadata = mocker.patch.object(twitter_client, "get_tweet_metadata", return_value={"id": "123456789"})

        assert await twitter_client.validate_tweet("https://twitter.com/user/status/123456789")
        mock_metadata.assert_called_once_with("123456789")

        assert not await twitter_client.validate_tweet("invalid_url")

    async def test_get_tweet_metadata(self, twitter_client: TwitterClient) -> None:
        """Test tweet metadata fetching.

        Args:
            twitter_client: Twitter client fixture
        """
        tweet_id = "123456789"
        api_response = {
            "data": {
                "id": tweet_id,
                "text": "Test tweet",
                "created_at": "2024-03-10T12:00:00Z",
                "author_id": "user123",
            },
            "includes": {
                "users": [{"id": "user123", "name": "Test User"}],
                "media": [{"url": "https://example.com/image.jpg"}],
            },
        }

        with aioresponses() as m:
            m.get(f"https://api.twitter.com/2/tweets/{tweet_id}", status=200, payload=api_response)

            async with twitter_client:
                metadata = await twitter_client.get_tweet_metadata(tweet_id)

            assert metadata["id"] == tweet_id
            assert metadata["author"] == "Test User"
            assert metadata["content"] == "Test tweet"
            assert len(metadata["media_urls"]) == 1

    async def test_get_thread_tweets(self, twitter_client: TwitterClient) -> None:
        """Test thread tweets fetching.

        Args:
            twitter_client: Twitter client fixture
        """
        thread_id = "123456789"
        first_tweet_response = {
            "data": {
                "id": thread_id,
                "text": "First tweet",
                "created_at": "2024-03-10T12:00:00Z",
                "author_id": "user123",
            },
            "includes": {"users": [{"id": "user123", "name": "Test User"}], "media": []},
        }

        thread_response = {
            "data": [
                {"id": "987654321", "text": "Reply tweet", "created_at": "2024-03-10T12:01:00Z", "author_id": "user123"}
            ],
            "includes": {"users": [{"id": "user123", "name": "Test User"}], "media": []},
        }

        with aioresponses() as m:
            m.get(f"https://api.twitter.com/2/tweets/{thread_id}", status=200, payload=first_tweet_response)
            m.get("https://api.twitter.com/2/tweets/search/recent", status=200, payload=thread_response)

            async with twitter_client:
                thread = await twitter_client.get_thread_tweets(thread_id)

            assert len(thread) == 2
            assert thread[0]["id"] == thread_id
            assert thread[1]["id"] == "987654321"
