# pyright: reportAttributeAccessIssue=false
"""Tests for Twitter cog functionality."""

from __future__ import annotations

import asyncio
import pathlib
import sys

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any

import discord
import discord.ext.test as dpytest

from discord.ext import commands
from loguru import logger

import pytest

from democracy_exe.chatbot.cogs.twitter import Twitter as TwitterCog
from democracy_exe.chatbot.cogs.twitter import TwitterError
from democracy_exe.utils.twitter_utils.types import TweetDownloadMode
from tests.tests_utils.last_ctx_cog import LastCtxCog


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

    from democracy_exe.chatbot.core.bot import DemocracyBot

# Constants for testing
TEST_TWEET_URL = "https://twitter.com/test/status/123456789"


@pytest.fixture(autouse=True)
async def twitter_cog(bot: DemocracyBot) -> TwitterCog:
    twitter_cog = TwitterCog(bot)
    await bot.add_cog(twitter_cog)
    dpytest.configure(bot)
    logger.info("Tests starting")
    return twitter_cog


@pytest.fixture
def mock_tweet_data() -> dict[str, Any]:
    """Create mock tweet data for testing.

    Returns:
        Mock tweet data dictionary
    """
    return {
        "success": True,
        "metadata": {
            "id": "123456789",
            "url": TEST_TWEET_URL,
            "author": "test_user",
            "content": "Test tweet content",
            "media_urls": [],
            "created_at": "2024-01-01",
        },
        "local_files": [],
        "error": None,
    }


@pytest.fixture
def mock_tweet_data_with_media(mock_tweet_data: dict[str, Any], tmp_path: pathlib.Path) -> dict[str, Any]:
    """Create mock tweet data with media files for testing.

    Args:
        mock_tweet_data: Base mock tweet data
        tmp_path: Temporary directory path

    Returns:
        Mock tweet data with media files
    """
    # Create a temporary file
    media_path = tmp_path / "gallery-dl" / "twitter" / "test_user"
    media_path.mkdir(parents=True)
    test_file = media_path / "test_media.mp4"
    test_file.write_text("test content")

    data = mock_tweet_data.copy()
    data["local_files"] = [str(test_file)]
    data["metadata"]["media_urls"] = ["https://test.com/media.mp4"]
    return data


@pytest.mark.asyncio
async def test_twitter_cog_init(twitter_cog: TwitterCog) -> None:
    """Test Twitter cog initialization.

    Args:
        twitter_cog: The Twitter cog instance
    """
    assert isinstance(twitter_cog, TwitterCog)
    assert isinstance(twitter_cog.bot, DemocracyBot)


@pytest.mark.asyncio
async def test_twitter_cog_on_ready(twitter_cog: TwitterCog, caplog: LogCaptureFixture) -> None:
    """Test Twitter cog on_ready event.

    Args:
        twitter_cog: The Twitter cog instance
        caplog: Pytest log capture fixture
    """
    await twitter_cog.on_ready()
    assert any(record.message == f"{type(twitter_cog).__name__} Cog ready." for record in caplog.records)


@pytest.mark.asyncio
async def test_twitter_cog_on_guild_join(
    twitter_cog: TwitterCog, test_guild: discord.Guild, caplog: LogCaptureFixture
) -> None:
    """Test Twitter cog on_guild_join event.

    Args:
        twitter_cog: The Twitter cog instance
        test_guild: Test guild fixture
        caplog: Pytest log capture fixture
    """
    await twitter_cog.on_guild_join(test_guild)
    assert any(record.message == f"Adding new guild to database: {test_guild.id}" for record in caplog.records)
    assert any(record.message == f"Successfully added guild {test_guild.id} to database" for record in caplog.records)


def assert_download_progress_embed(embed: discord.Embed, url: str, mode: TweetDownloadMode) -> None:
    """Assert download progress embed is correct.

    Args:
        embed: Discord embed to check
        url: Tweet URL
        mode: Download mode
    """
    assert embed.title == "Download in Progress"
    assert url in embed.description
    assert mode in embed.description.lower()


def assert_error_embed(embed: discord.Embed, error_msg: str) -> None:
    """Assert error embed is correct.

    Args:
        embed: Discord embed to check
        error_msg: Expected error message
    """
    assert embed.title == "Error"
    assert error_msg in embed.description
    assert embed.color == discord.Color.red()


def assert_success_message(message: discord.Message) -> None:
    """Assert success message is correct.

    Args:
        message: Discord message to check
    """
    assert "Download complete!" in message.content
