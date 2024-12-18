# pyright: reportAttributeAccessIssue=false
"""Tests for Twitter cog functionality."""

from __future__ import annotations

import asyncio
import pathlib
import sys

from collections.abc import AsyncGenerator, Generator
from lib2to3.pytree import _Results
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
TEST_TWEET_URL = "https://x.com/bancodevideo/status/1699925133194858974"

# @pytest.mark.vcronly()
# @pytest.mark.default_cassette("test_new_dev_questions_success.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "body", "headers"],
#     ignore_localhost=False,
# )
# =========================== short test summary info ============================
# FAILED tests/cogs/test_twitter_cog.py::test_download_tweet_success - asyncio.queues.QueueEmpty
# FAILED tests/cogs/test_twitter_cog.py::test_download_tweet_with_media - asyncio.queues.QueueEmpty
# FAILED tests/cogs/test_twitter_cog.py::test_download_tweet_failure - AssertionError: assert 'Download in Progress' == 'Error'
# FAILED tests/cogs/test_twitter_cog.py::test_download_tweet_invalid_url - AssertionError: assert 'Download in Progress' == 'Error'
# FAILED tests/cogs/test_twitter_cog.py::test_download_tweet_missing_url - discord.ext.commands.errors.MissingRequiredArgument: url is a required argu...
# FAILED tests/cogs/test_twitter_cog.py::test_download_thread_success - asyncio.queues.QueueEmpty
# FAILED tests/cogs/test_twitter_cog.py::test_download_card_success - asyncio.queues.QueueEmpty
# FAILED tests/cogs/test_twitter_cog.py::test_info_command_success - discord.ext.commands.errors.CommandInvokeError: Command raised an exception...
# FAILED tests/cogs/test_twitter_cog.py::test_info_command_failure - discord.ext.commands.errors.CommandInvokeError: Command raised an exception...
# FAILED tests/cogs/test_twitter_cog.py::test_missing_permissions - discord.ext.commands.errors.MissingPermissions: You are missing Manage Serv...
# FAILED tests/cogs/test_twitter_cog.py::test_command_cooldown - discord.ext.commands.errors.CommandOnCooldown: You are on cooldown. Try aga...
# FAILED tests/cogs/test_twitter_cog.py::test_download_tweet_network_error - AssertionError: assert 'Download in Progress' == 'Error'
# FAILED tests/cogs/test_twitter_cog.py::test_download_tweet_rate_limit - AssertionError: assert 'Download in Progress' == 'Error'

# _Results (5.64s):
#        6 passed
#       13 failed
#          - tests/cogs/test_twitter_cog.py:176 test_download_tweet_success
#          - tests/cogs/test_twitter_cog.py:205 test_download_tweet_with_media
#          - tests/cogs/test_twitter_cog.py:235 test_download_tweet_failure
#          - tests/cogs/test_twitter_cog.py:256 test_download_tweet_invalid_url
#          - tests/cogs/test_twitter_cog.py:272 test_download_tweet_missing_url
#          - tests/cogs/test_twitter_cog.py:288 test_download_thread_success
#          - tests/cogs/test_twitter_cog.py:317 test_download_card_success
#          - tests/cogs/test_twitter_cog.py:379 test_info_command_success
#          - tests/cogs/test_twitter_cog.py:404 test_info_command_failure
#          - tests/cogs/test_twitter_cog.py:430 test_missing_permissions
#          - tests/cogs/test_twitter_cog.py:450 test_command_cooldown
#          - tests/cogs/test_twitter_cog.py:472 test_download_tweet_network_error
#          - tests/cogs/test_twitter_cog.py:493 test_download_tweet_rate_limit


@pytest.fixture(autouse=True)
async def twitter_cog(bot: DemocracyBot) -> TwitterCog:
    """Create and configure Twitter cog for testing.

    Args:
        bot: The Discord bot instance

    Returns:
        Configured TwitterCog instance
    """
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


@pytest.mark.asyncio
async def test_twitter_cog_init(twitter_cog: TwitterCog) -> None:
    """Test Twitter cog initialization.

    Args:
        twitter_cog: The Twitter cog instance
    """
    assert isinstance(twitter_cog, TwitterCog)
    assert isinstance(twitter_cog.bot, commands.Bot)


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


@pytest.mark.asyncio
async def test_download_tweet_success(
    twitter_cog: TwitterCog, mocker: MockerFixture, mock_tweet_data: dict[str, Any]
) -> None:
    """Test successful tweet download.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
        mock_tweet_data: Mock tweet data
    """
    # Mock download_tweet function
    mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
    mock_download.return_value = mock_tweet_data

    # Send download command
    await dpytest.message("?tweet download " + TEST_TWEET_URL)

    # Verify progress message
    messages = dpytest.get_message()
    assert messages is not None
    assert_download_progress_embed(messages.embeds[0], TEST_TWEET_URL, "single")

    # Verify success message
    messages = dpytest.get_message()
    assert messages is not None
    assert "Download complete!" in messages.content


@pytest.mark.asyncio
async def test_download_tweet_with_media(
    twitter_cog: TwitterCog, mocker: MockerFixture, mock_tweet_data_with_media: dict[str, Any]
) -> None:
    """Test tweet download with media files.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
        mock_tweet_data_with_media: Mock tweet data with media
    """
    # Mock download_tweet function
    mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
    mock_download.return_value = mock_tweet_data_with_media

    # Send download command
    await dpytest.message("?tweet download " + TEST_TWEET_URL)

    # Verify progress message
    messages = dpytest.get_message()
    assert messages is not None
    assert_download_progress_embed(messages.embeds[0], TEST_TWEET_URL, "single")

    # Verify success message and file upload
    messages = dpytest.get_message()
    assert messages is not None
    assert "Download complete!" in messages.content
    assert len(messages.attachments) > 0


@pytest.mark.asyncio
async def test_download_tweet_failure(twitter_cog: TwitterCog, mocker: MockerFixture) -> None:
    """Test tweet download failure handling.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
    """
    # Mock download_tweet function to fail
    mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
    mock_download.return_value = {"success": False, "error": "Download failed", "metadata": None, "local_files": []}

    # Send download command
    await dpytest.message("?tweet download " + TEST_TWEET_URL)

    # Verify error message
    messages = dpytest.get_message()
    assert messages is not None
    assert_error_embed(messages.embeds[0], "Download failed")


@pytest.mark.asyncio
async def test_download_tweet_invalid_url(twitter_cog: TwitterCog) -> None:
    """Test tweet download with invalid URL.

    Args:
        twitter_cog: The Twitter cog instance
    """
    # Send download command with invalid URL
    await dpytest.message("?tweet download invalid_url")

    # Verify error message
    messages = dpytest.get_message()
    assert messages is not None
    assert_error_embed(messages.embeds[0], "Failed to download tweet")


@pytest.mark.asyncio
async def test_download_tweet_missing_url(twitter_cog: TwitterCog) -> None:
    """Test tweet download with missing URL.

    Args:
        twitter_cog: The Twitter cog instance
    """
    # Send download command without URL
    await dpytest.message("?tweet download")

    # Verify help message
    messages = dpytest.get_message()
    assert messages is not None
    assert "HELP_MESSAGE" in messages.content


@pytest.mark.asyncio
async def test_download_thread_success(
    twitter_cog: TwitterCog, mocker: MockerFixture, mock_tweet_data: dict[str, Any]
) -> None:
    """Test successful thread download.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
        mock_tweet_data: Mock tweet data
    """
    # Mock download_tweet function
    mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
    mock_download.return_value = mock_tweet_data

    # Send thread download command
    await dpytest.message("?tweet thread " + TEST_TWEET_URL)

    # Verify progress message
    messages = dpytest.get_message()
    assert messages is not None
    assert_download_progress_embed(messages.embeds[0], TEST_TWEET_URL, "thread")

    # Verify success message
    messages = dpytest.get_message()
    assert messages is not None
    assert "Download complete!" in messages.content


@pytest.mark.asyncio
async def test_download_card_success(
    twitter_cog: TwitterCog, mocker: MockerFixture, mock_tweet_data: dict[str, Any]
) -> None:
    """Test successful card download.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
        mock_tweet_data: Mock tweet data
    """
    # Mock download_tweet function
    mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
    mock_download.return_value = mock_tweet_data

    # Send card download command
    await dpytest.message("?tweet card " + TEST_TWEET_URL)

    # Verify progress message
    messages = dpytest.get_message()
    assert messages is not None
    assert_download_progress_embed(messages.embeds[0], TEST_TWEET_URL, "card")

    # Verify success message
    messages = dpytest.get_message()
    assert messages is not None
    assert "Download complete!" in messages.content


@pytest.mark.asyncio
async def test_cleanup_temp_dir(twitter_cog: TwitterCog, tmp_path: pathlib.Path) -> None:
    """Test temporary directory cleanup.

    Args:
        twitter_cog: The Twitter cog instance
        tmp_path: Temporary directory path
    """
    # Create test directory structure
    gallery_dl_dir = tmp_path / "gallery-dl"
    gallery_dl_dir.mkdir()
    test_file = gallery_dl_dir / "test.txt"
    test_file.write_text("test")

    # Call cleanup
    twitter_cog._cleanup_temp_dir(str(test_file))

    # Verify directory is cleaned up
    assert not gallery_dl_dir.exists()


@pytest.mark.asyncio
async def test_cleanup_temp_dir_nonexistent(twitter_cog: TwitterCog) -> None:
    """Test cleanup with nonexistent directory.

    Args:
        twitter_cog: The Twitter cog instance
    """
    # Call cleanup with nonexistent path
    twitter_cog._cleanup_temp_dir("/nonexistent/path")
    # Should not raise any exceptions


@pytest.mark.asyncio
async def test_info_command_success(
    twitter_cog: TwitterCog, mocker: MockerFixture, mock_tweet_data: dict[str, Any]
) -> None:
    """Test successful info command execution.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
        mock_tweet_data: Mock tweet data
    """
    # Mock download_tweet function
    mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
    mock_download.return_value = mock_tweet_data

    # Send info command
    await dpytest.message("?tweet info " + TEST_TWEET_URL)

    # Verify info embed
    messages = dpytest.get_message()
    assert messages is not None
    assert messages.embeds[0].title == "Tweet Information"
    assert TEST_TWEET_URL in messages.embeds[0].description


@pytest.mark.asyncio
async def test_info_command_failure(twitter_cog: TwitterCog, mocker: MockerFixture) -> None:
    """Test info command failure handling.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
    """
    # Mock download_tweet function to fail
    mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
    mock_download.return_value = {
        "success": False,
        "error": "Failed to fetch info",
        "metadata": None,
        "local_files": [],
    }

    # Send info command
    await dpytest.message("?tweet info " + TEST_TWEET_URL)

    # Verify error embed
    messages = dpytest.get_message()
    assert messages is not None
    assert_error_embed(messages.embeds[0], "Failed to fetch info")


@pytest.mark.asyncio
async def test_missing_permissions(twitter_cog: TwitterCog, mocker: MockerFixture) -> None:
    """Test handling of missing permissions.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
    """
    # Mock commands.MissingPermissions error
    mocker.patch.object(twitter_cog.download, "can_run", side_effect=commands.MissingPermissions(["manage_server"]))

    # Send download command
    await dpytest.message("?tweet download " + TEST_TWEET_URL)

    # Verify permission error message
    messages = dpytest.get_message()
    assert messages is not None
    assert "MANAGE SERVER" in messages.embeds[0].description


@pytest.mark.asyncio
async def test_command_cooldown(twitter_cog: TwitterCog, mocker: MockerFixture) -> None:
    """Test command cooldown handling.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
    """
    # Mock commands.CommandOnCooldown error
    mocker.patch.object(
        twitter_cog.download, "can_run", side_effect=commands.CommandOnCooldown(commands.BucketType.default, 5.0, 0)
    )

    # Send download command
    await dpytest.message("?tweet download " + TEST_TWEET_URL)

    # Verify cooldown message
    messages = dpytest.get_message()
    assert messages is not None
    assert "cooldown" in messages.embeds[0].description.lower()


@pytest.mark.asyncio
async def test_download_tweet_network_error(twitter_cog: TwitterCog, mocker: MockerFixture) -> None:
    """Test handling of network errors during download.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
    """
    # Mock download_tweet function to raise network error
    mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
    mock_download.side_effect = ConnectionError("Network error")

    # Send download command
    await dpytest.message("?tweet download " + TEST_TWEET_URL)

    # Verify error message
    messages = dpytest.get_message()
    assert messages is not None
    assert_error_embed(messages.embeds[0], "Network error")


@pytest.mark.asyncio
async def test_download_tweet_rate_limit(twitter_cog: TwitterCog, mocker: MockerFixture) -> None:
    """Test handling of rate limit errors.

    Args:
        twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
    """
    # Mock download_tweet function to simulate rate limit
    mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
    mock_download.return_value = {"success": False, "error": "Rate limit exceeded", "metadata": None, "local_files": []}

    # Send download command
    await dpytest.message("?tweet download " + TEST_TWEET_URL)

    # Verify rate limit error message
    messages = dpytest.get_message()
    assert messages is not None
    assert_error_embed(messages.embeds[0], "Rate limit exceeded")


@pytest.mark.asyncio
async def test_cleanup_temp_dir_permission_error(
    twitter_cog: TwitterCog, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Test cleanup with permission error.

    Args:
        twitter_cog: The Twitter cog instance
        tmp_path: Temporary directory path
        mocker: Pytest mocker fixture
    """
    # Create test directory structure
    gallery_dl_dir = tmp_path / "gallery-dl"
    gallery_dl_dir.mkdir()
    test_file = gallery_dl_dir / "test.txt"
    test_file.write_text("test")

    # Mock shutil.rmtree to raise permission error
    mock_rmtree = mocker.patch("shutil.rmtree")
    mock_rmtree.side_effect = PermissionError("Permission denied")

    # Call cleanup
    twitter_cog._cleanup_temp_dir(str(test_file))

    # Verify directory still exists (cleanup failed gracefully)
    assert gallery_dl_dir.exists()
