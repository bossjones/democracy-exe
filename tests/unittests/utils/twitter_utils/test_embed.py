"""Tests for Discord embed utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import discord

import pytest

from democracy_exe.utils.twitter_utils.embed import (
    BLUE,
    GOLD,
    RED,
    TweetMetadata,
    create_card_embed,
    create_download_progress_embed,
    create_error_embed,
    create_info_embed,
    create_thread_embed,
    create_tweet_embed,
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
TEST_CREATED_AT = "2024-01-01T12:00:00Z"
TEST_MEDIA_URLS = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
]
TEST_CARD_URL = "https://example.com"
TEST_CARD_DESCRIPTION = "Card description"
TEST_CARD_IMAGE = "https://example.com/image.jpg"


@pytest.fixture
def sample_metadata() -> TweetMetadata:
    """Create sample tweet metadata.

    Returns:
        Sample tweet metadata
    """
    return {
        "id": TEST_TWEET_ID,
        "author": TEST_AUTHOR,
        "content": TEST_CONTENT,
        "created_at": TEST_CREATED_AT,
        "url": TEST_URL,
        "retweet_count": 10,
        "like_count": 20,
        "reply_count": 5,
    }


@pytest.fixture
def sample_thread_metadata(sample_metadata: TweetMetadata) -> list[TweetMetadata]:
    """Create sample thread metadata.

    Args:
        sample_metadata: Sample tweet metadata

    Returns:
        List of tweet metadata for thread
    """
    thread = []
    for i in range(3):
        tweet = sample_metadata.copy()
        tweet.update({
            "id": f"{tweet['id']}{i}",
            "content": f"Thread tweet {i + 1}",
        })
        thread.append(tweet)
    return thread


@pytest.fixture
def sample_card_metadata(sample_metadata: TweetMetadata) -> TweetMetadata:
    """Create sample card metadata.

    Args:
        sample_metadata: Sample tweet metadata

    Returns:
        Tweet metadata with card
    """
    metadata = sample_metadata.copy()
    metadata.update({
        "card_url": TEST_CARD_URL,
        "card_description": TEST_CARD_DESCRIPTION,
        "card_image": TEST_CARD_IMAGE,
    })
    return metadata


class TestTweetEmbed:
    """Tests for tweet embed creation."""

    def test_create_basic_embed(self, sample_metadata: TweetMetadata) -> None:
        """Test creation of basic tweet embed.

        Args:
            sample_metadata: Sample tweet metadata
        """
        embed = create_tweet_embed(sample_metadata)

        assert isinstance(embed, discord.Embed)
        assert embed.color == BLUE
        assert embed.author.name == sample_metadata["author"]
        assert embed.description == sample_metadata["content"]
        assert any(f.name == "Created" for f in embed.fields)
        assert any(f.name == "URL" for f in embed.fields)

    def test_create_with_media(self, sample_metadata: TweetMetadata) -> None:
        """Test creation of tweet embed with media.

        Args:
            sample_metadata: Sample tweet metadata
        """
        metadata = sample_metadata.copy()
        metadata["media_urls"] = TEST_MEDIA_URLS

        embed = create_tweet_embed(metadata)

        assert isinstance(embed, discord.Embed)
        assert any(f.name == "Media URLs" for f in embed.fields)
        media_field = next(f for f in embed.fields if f.name == "Media")
        assert all(url in media_field.value for url in metadata["media_urls"])

    def test_create_without_optional_fields(self) -> None:
        """Test creation of tweet embed with minimal metadata."""
        minimal_metadata: TweetMetadata = {
            "id": TEST_TWEET_ID,
            "author": TEST_AUTHOR,
            "content": TEST_CONTENT,
            "created_at": TEST_CREATED_AT,
            "url": TEST_URL,
        }

        embed = create_tweet_embed(minimal_metadata)

        assert isinstance(embed, discord.Embed)
        assert embed.author.name == minimal_metadata["author"]
        assert embed.description == minimal_metadata["content"]


class TestThreadEmbed:
    """Tests for thread embed creation."""

    def test_create_thread(self, sample_thread_metadata: list[TweetMetadata]) -> None:
        """Test creation of thread embed.

        Args:
            sample_thread_metadata: Sample thread metadata
        """
        embed = create_thread_embed(sample_thread_metadata)

        assert isinstance(embed, discord.Embed)
        assert embed.color == BLUE
        assert embed.author.name == sample_thread_metadata[0]["author"]
        assert "Thread Length" in [f.name for f in embed.fields]
        assert str(len(sample_thread_metadata)) in next(f.value for f in embed.fields if f.name == "Thread Length")
        assert all(f"Tweet {i + 1}:" in embed.description for i in range(len(sample_thread_metadata)))

    def test_create_empty_thread(self) -> None:
        """Test creation of thread embed with empty thread."""
        embed = create_thread_embed([])

        assert isinstance(embed, discord.Embed)
        assert embed.title == "Empty Thread"
        assert embed.description == "No tweets found"

    def test_create_single_tweet_thread(self, sample_metadata: TweetMetadata) -> None:
        """Test creation of thread embed with single tweet.

        Args:
            sample_metadata: Sample tweet metadata
        """
        embed = create_thread_embed([sample_metadata])

        assert isinstance(embed, discord.Embed)
        assert embed.author.name == sample_metadata["author"]
        assert "Thread Length" in [f.name for f in embed.fields]
        assert "1" in next(f.value for f in embed.fields if f.name == "Thread Length")


class TestCardEmbed:
    """Tests for card embed creation."""

    def test_create_card(self, sample_card_metadata: TweetMetadata) -> None:
        """Test creation of card embed.

        Args:
            sample_card_metadata: Sample card metadata
        """
        embed = create_card_embed(sample_card_metadata)

        assert isinstance(embed, discord.Embed)
        assert embed.color == BLUE
        assert embed.author.name == sample_card_metadata["author"]
        assert embed.description == sample_card_metadata["content"]
        assert any(f.name == "Card URL" for f in embed.fields)
        assert any(f.name == "Description" for f in embed.fields)
        assert embed.image.url == sample_card_metadata["card_image"]

    def test_create_card_without_image(self, sample_card_metadata: TweetMetadata) -> None:
        """Test creation of card embed without image.

        Args:
            sample_card_metadata: Sample card metadata
        """
        metadata = sample_card_metadata.copy()
        del metadata["card_image"]

        embed = create_card_embed(metadata)

        assert isinstance(embed, discord.Embed)
        assert embed.image.url is None


class TestInfoEmbed:
    """Tests for info embed creation."""

    def test_create_info(self, sample_metadata: TweetMetadata) -> None:
        """Test creation of info embed.

        Args:
            sample_metadata: Sample tweet metadata
        """
        embed = create_info_embed(sample_metadata)

        assert isinstance(embed, discord.Embed)
        assert embed.color == BLUE
        assert embed.description == sample_metadata["content"]
        assert any(f.name == "ID" for f in embed.fields)
        assert any(f.name == "Author" for f in embed.fields)
        assert any(f.name == "Created" for f in embed.fields)
        assert any(f.name == "URL" for f in embed.fields)
        assert any(f.name == "Retweet Count" for f in embed.fields)
        assert any(f.name == "Like Count" for f in embed.fields)
        assert any(f.name == "Reply Count" for f in embed.fields)

    def test_create_info_with_media(self, sample_metadata: TweetMetadata) -> None:
        """Test creation of info embed with media.

        Args:
            sample_metadata: Sample tweet metadata
        """
        metadata = sample_metadata.copy()
        metadata["media_urls"] = TEST_MEDIA_URLS

        embed = create_info_embed(metadata)

        assert isinstance(embed, discord.Embed)
        assert any(f.name == "Media URLs" for f in embed.fields)
        media_field = next(f for f in embed.fields if f.name == "Media URLs")
        assert all(url in media_field.value for url in metadata["media_urls"])


class TestProgressEmbed:
    """Tests for progress embed creation."""

    @pytest.mark.parametrize("mode", ["single", "thread", "card"])
    def test_create_progress(self, mode: TweetDownloadMode) -> None:
        """Test creation of download progress embed.

        Args:
            mode: Download mode to test
        """
        embed = create_download_progress_embed(TEST_URL, mode)

        assert isinstance(embed, discord.Embed)
        assert embed.color == GOLD
        assert embed.title == "Tweet Download"
        assert TEST_URL in embed.description
        assert mode in embed.description
        assert embed.footer.text == "Please wait..."


class TestErrorEmbed:
    """Tests for error embed creation."""

    def test_create_error(self) -> None:
        """Test creation of error embed."""
        error_message = "Test error message"
        embed = create_error_embed(error_message)

        assert isinstance(embed, discord.Embed)
        assert embed.color == RED
        assert embed.title == "Error"
        assert embed.description == error_message

    def test_create_error_with_empty_message(self) -> None:
        """Test creation of error embed with empty message."""
        embed = create_error_embed("")

        assert isinstance(embed, discord.Embed)
        assert embed.description == ""
