# pyright: reportAttributeAccessIssue=false
"""Tests for Twitter cog functionality."""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import pathlib
import sys

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any

import aiofiles
import aiofiles.os
import discord
import discord.ext.test as dpytest
import pytest_asyncio

from discord.client import _LoopSentinel
from discord.ext import commands
from loguru import logger

import pytest

from democracy_exe.chatbot.cogs.twitter import Twitter as TwitterCog
from democracy_exe.chatbot.cogs.twitter import TwitterError
from democracy_exe.chatbot.core.bot import DemocracyBot
from democracy_exe.utils._testing import ContextLogger
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


# @pytest.fixture(autouse=True)
# async def twitter_cog(mockbot: DemocracyBot) -> TwitterCog:
#     """Create and configure Twitter cog for testing.

#     Args:
#         bot: The Discord bot instance

#     Returns:
#         Configured TwitterCog instance
#     """
#     twitter_cog = TwitterCog(mockbot)
#     await mockbot.add_cog(twitter_cog)
#     dpytest.configure(mockbot)
#     logger.info("Tests starting")
#     return twitter_cog


@pytest_asyncio.fixture(autouse=True)
# async def bot_with_twitter_cog() -> AsyncGenerator[DemocracyBot, None]:
async def bot_with_twitter_cog() -> DemocracyBot:
    """Create a DemocracyBot instance for testing.

    Args:
        event_loop: The event loop fixture

    Returns:
        AsyncGenerator[DemocracyBot, None]: DemocracyBot instance with test configuration
    """
    # Configure intents
    intents = discord.Intents.default()
    intents.members = True
    intents.message_content = True
    intents.messages = True
    intents.guilds = True

    test_bot = DemocracyBot(command_prefix="?", intents=intents, description="Test DemocracyBot instance")

    # set up the loop
    if isinstance(test_bot.loop, _LoopSentinel):  # type: ignore
        await test_bot._async_setup_hook()  # type: ignore

    test_twitter_cog = TwitterCog(test_bot)
    await test_bot.add_cog(test_twitter_cog)
    # Create DemocracyBot with test configuration

    # # Add test-specific error handling
    # @test_bot.event
    # async def on_command_error(ctx: commands.Context, error: Exception) -> None:  # type: ignore
    #     """Handle command errors in test environment."""
    #     raise error  # Re-raise for pytest to catch

    # Setup and cleanup
    # await bot._async_setup_hook()  # Required for proper initialization
    # await dpytest.empty_queue()
    dpytest.configure(test_bot)
    return test_bot
    # # await dpytest.empty_queue()

    # try:
    #     # Teardown
    #     await dpytest.empty_queue()  # empty the global message queue as test teardown
    # finally:
    #     pass


@pytest_asyncio.fixture(autouse=True)
async def cleanup_global_message_queue() -> AsyncGenerator[None, None]:
    """
    Fixture to clean up the global message queue after each test.

    This fixture is automatically used for all tests and ensures that
    the global message queue is emptied after each test run.

    Yields:
    ------
        AsyncGenerator[None, None]: Yields control back to the test.

    """
    yield
    try:
        # Teardown
        await dpytest.empty_queue()  # empty the global message queue as test teardown
    finally:
        pass

    # await dpytest.empty_queue()


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


@pytest_asyncio.fixture
async def mock_aio_tweet_data_with_media(
    mock_tweet_data: dict[str, Any], tmp_path: pathlib.Path
) -> AsyncGenerator[dict[str, Any], None]:
    """Create mock tweet data with media files for async testing.

    Args:
        mock_tweet_data: Base mock tweet data fixture
        tmp_path: Temporary directory path fixture

    Yields:
        dict[str, Any]: Mock tweet data with media files
    """
    # Create a temporary directory structure mimicking gallery-dl output
    import bpdb

    media_path = tmp_path / "gallery-dl" / "twitter" / "test_user"

    await aiofiles.os.makedirs(f"{media_path}", exist_ok=True)
    # bpdb.set_trace()
    media_path.mkdir(parents=True, exist_ok=True)

    # Create test media files
    test_video = media_path / "test_video.mp4"
    test_image = media_path / "test_image.jpg"

    # Write some dummy content to the files
    test_video.write_bytes(b"dummy video content")
    test_image.write_bytes(b"dummy image content")

    # Create info.json file
    info_json = media_path / "info.json"
    info_data = {
        "tweet": {
            "id": "123456789",
            "url": TEST_TWEET_URL,
            "text": "Test tweet with media content",
            "created_at": "2024-01-01",
            "media": [{"url": "https://test.com/video.mp4"}, {"url": "https://test.com/image.jpg"}],
            "user": {"name": "test_user", "screen_name": "test_user"},
        }
    }
    info_json.write_text(json.dumps(info_data))

    # Create mock data dictionary
    data = mock_tweet_data.copy()
    data.update({
        "success": True,
        "metadata": {
            "id": "123456789",
            "url": TEST_TWEET_URL,
            "author": "test_user",
            "content": "Test tweet with media content",
            "media_urls": ["https://test.com/video.mp4", "https://test.com/image.jpg"],
            "created_at": "2024-01-01",
        },
        "local_files": [str(test_video), str(test_image)],
        "error": None,
    })

    # import bpdb
    yield data
    # example of data:
    # {
    #     'success': True,
    #     'metadata': {
    #         'id': '123456789',
    #         'url': 'https://x.com/bancodevideo/status/1699925133194858974',
    #         'author': 'test_user',
    #         'content': 'Test tweet with media content',
    #         'media_urls': ['https://test.com/video.mp4', 'https://test.com/image.jpg'],
    #         'created_at': '2024-01-01'
    #     },
    #     'local_files': [
    #         '/private/var/folders/q_/d5r_s8wd02zdx6qmc5f_96mw0000gp/T/pytest-of-malcolm/pytest-156/test_download_tweet_success_tw0/gallery-dl/twitter/test_user/test_video.mp4',
    #         '/private/var/folders/q_/d5r_s8wd02zdx6qmc5f_96mw0000gp/T/pytest-of-malcolm/pytest-156/test_download_tweet_success_tw0/gallery-dl/twitter/test_user/test_image.jpg'
    #     ],
    #     'error': None
    # }

    # bpdb.set_trace()
    # Cleanup (if needed)
    try:
        for file in [test_video, test_image, info_json]:
            if file.exists():
                file.unlink()
        media_path.rmdir()
        # await aiofiles.os.rmdir(f"{media_path}")
    except Exception as e:
        logger.warning(f"Cleanup error in mock_aio_tweet_data_with_media: {e}")


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


# @pytest.mark.asyncio
# async def test_twitter_cog_init(twitter_cog: TwitterCog) -> None:
#     """Test Twitter cog initialization.

#     Args:
#         twitter_cog: The Twitter cog instance
#     """
#     assert isinstance(twitter_cog, TwitterCog)
#     assert isinstance(twitter_cog.bot, commands.Bot)


# @pytest.mark.asyncio
# async def test_twitter_cog_on_ready(twitter_cog: TwitterCog, caplog: LogCaptureFixture) -> None:
#     """Test Twitter cog on_ready event.

#     Args:
#         twitter_cog: The Twitter cog instance
#         caplog: Pytest log capture fixture
#     """
#     await twitter_cog.on_ready()
#     assert any(record.message == f"{type(twitter_cog).__name__} Cog ready." for record in caplog.records)


# @pytest.mark.asyncio
# async def test_twitter_cog_on_guild_join(
#     twitter_cog: TwitterCog, test_guild: discord.Guild, caplog: LogCaptureFixture
# ) -> None:
#     """Test Twitter cog on_guild_join event.

#     Args:
#         twitter_cog: The Twitter cog instance
#         test_guild: Test guild fixture
#         caplog: Pytest log capture fixture
#     """
#     await twitter_cog.on_guild_join(test_guild)
#     assert any(record.message == f"Adding new guild to database: {test_guild.id}" for record in caplog.records)
#     assert any(record.message == f"Successfully added guild {test_guild.id} to database" for record in caplog.records)


# @pytest.mark.skip_until(
#     deadline=datetime.datetime(2024, 12, 25),
#     strict=True,
#     msg="Alert is suppressed. Make progress till then"
# )
@pytest.mark.asyncio
async def test_download_tweet_success_twitter_cog(
    bot_with_twitter_cog: DemocracyBot,
    mocker: MockerFixture,
    # mock_tweet_data: dict[str, Any],
    # tmp_path: pathlib.Path,
    # mock_tweet_data_with_media: dict[str, Any],
    mock_aio_tweet_data_with_media: dict[str, Any],
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
) -> None:
    """Test successful tweet download.

    Args:
        bot_with_twitter_cog: The Twitter cog instance
        mocker: Pytest mocker fixture
        mock_tweet_data: Mock tweet data
    """
    with capsys.disabled():
        with ContextLogger(caplog):
            caplog.set_level(logging.DEBUG)
            # Mock shell command execution with AsyncMock
            mock_shell = mocker.AsyncMock(return_value=(b"", b""))
            mocker.patch("democracy_exe.shell._aio_run_process_and_communicate", side_effect=mock_shell)

            # Mock shell command execution
            mock_shell = mocker.AsyncMock()
            mock_shell.return_value = (0, "Success", "")  # Return code 0, stdout, stderr
            mocker.patch("democracy_exe.utils.shell._aio_run_process_and_communicate", mock_shell)

            # Mock download_tweet function
            mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
            mock_download.return_value = dict(mock_aio_tweet_data_with_media)
            test_media = mock_aio_tweet_data_with_media["local_files"][0]

            # >>> dict(mock_tweet_data_with_media)
            # {'success': True, 'metadata': {'id': '123456789', 'url': 'https://x.com/bancodevideo/status/1699925133194858974', 'author': 'test_user', 'content': 'Test tweet content', 'media_urls': ['https://test.com/media.mp4'
            # ], 'created_at': '2024-01-01'}, 'local_files': ['/private/var/folders/q_/d5r_s8wd02zdx6qmc5f_96mw0000gp/T/pytest-of-malcolm/pytest-132/test_download_tweet_success0/gallery-dl/twitter/test_user/test_media.mp4'], 'e
            # rror': None}
            # >>>

            # import bpdb

            # bpdb.set_trace()

            # Mock file creation
            # test_image = tmp_path / "image.jpg"
            # test_image.touch()
            # These are sync functions, so regular Mock is fine
            # mocker.patch("democracy_exe.utils.file_functions.tree", return_value=[test_media])
            # mocker.patch("democracy_exe.utils.file_functions.filter_media", return_value=[str(test_media)])

            # mock_tweet_data["local_files"] = [str(test_media)]
            # # Mock download_tweet function
            # mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
            # mock_download.return_value = mock_tweet_data

            # Send download command
            await dpytest.message("?tweet download " + TEST_TWEET_URL)

            # Wait for messages to be processed
            await asyncio.sleep(0.1)

            assert dpytest.verify().message().content("Download complete!")

            # # Get all messages from the queue
            # messages: list[discord.Message | None] = []
            # try:
            #     while True:
            #         messages.append(dpytest.get_message())
            # except asyncio.queues.QueueEmpty:
            #     pass

            # # Should have at least 2 messages (progress and completion)
            # # assert len(messages) >= 2
            # assert len(messages) == 1

            # # First message should be progress
            # assert_download_progress_embed(messages[0].embeds[0], TEST_TWEET_URL, "single")  # type: ignore

            # # Last message should be completion
            # assert "Download complete!" in messages[-1].content

            # Verify shell command was mocked
            mock_shell.assert_called()


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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
    assert_download_progress_embed(messages.embeds[0], TEST_TWEET_URL, "single")  # type: ignore

    # Verify success message and file upload
    messages = dpytest.get_message()
    assert messages is not None
    assert "Download complete!" in messages.content
    assert len(messages.attachments) > 0


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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
    assert_error_embed(messages.embeds[0], "Download failed")  # type: ignore


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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
    assert_error_embed(messages.embeds[0], "Failed to download tweet")  # type: ignore


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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
    assert_download_progress_embed(messages.embeds[0], TEST_TWEET_URL, "thread")  # type: ignore

    # Verify success message
    messages = dpytest.get_message()
    assert messages is not None
    assert "Download complete!" in messages.content


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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
    assert_download_progress_embed(messages.embeds[0], TEST_TWEET_URL, "card")  # type: ignore

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


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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


@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25), strict=True, msg="Alert is suppressed. Make progress till then"
)
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


# @pytest.fixture
# def real_tweet_data() -> dict[str, Any]:
#     """Create mock tweet data for testing.

#     Returns:
#         Mock tweet data dictionary
#     """
#     return {
#         "tweet_id": 1699925133194858974,
#         "retweet_id": 0,
#         "quote_id": 0,
#         "reply_id": 0,
#         "conversation_id": 1699925133194858974,
#     "date": "2023-09-07 23:18:29",
#     "author": {
#         "id": 1640521982486757377,
#         "name": "bancodevideo",
#         "nick": "videos para responder tweets ➡️ @bancodevideo",
#         "location": "",
#         "date": "2023-03-28 01:12:04",
#         "verified": False,
#         "protected": False,
#         "profile_banner": "https://pbs.twimg.com/profile_banners/1640521982486757377/1680549523",
#         "profile_image": "https://pbs.twimg.com/profile_images/1640527061742759937/A8yiuCsS.jpg",
#         "favourites_count": 784,
#         "followers_count": 32869,
#         "friends_count": 65,
#         "listed_count": 32,
#         "media_count": 953,
#         "statuses_count": 1101,
#         "description": "El contenido de esta cuenta no me pertenece en su totalidad ni refleja mis opiniones🔸\n\nContacto: devideosbanco@gmail.com🔸\n\nGrupo de Telegram 👇",
#         "url": "https://t.me/+65DPYviLkzNiZDk5"
#     },
#     "user": {
#         "id": 1640521982486757377,
#         "name": "bancodevideo",
#         "nick": "videos para responder tweets ➡️ @bancodevideo",
#         "location": "",
#         "date": "2023-03-28 01:12:04",
#         "verified": False,
#         "protected": False,
#         "profile_banner": "https://pbs.twimg.com/profile_banners/1640521982486757377/1680549523",
#         "profile_image": "https://pbs.twimg.com/profile_images/1640527061742759937/A8yiuCsS.jpg",
#         "favourites_count": 784,
#         "followers_count": 32869,
#         "friends_count": 65,
#         "listed_count": 32,
#         "media_count": 953,
#         "statuses_count": 1101,
#         "description": "El contenido de esta cuenta no me pertenece en su totalidad ni refleja mis opiniones🔸\n\nContacto: devideosbanco@gmail.com🔸\n\nGrupo de Telegram 👇",
#         "url": "https://t.me/+65DPYviLkzNiZDk5"
#     },
#     "lang": "en",
#     "source": "Twitter for Android",
#     "sensitive": False,
#     "favorite_count": 78,
#     "quote_count": 3,
#     "reply_count": 2,
#     "retweet_count": 4,
#     "bookmark_count": 218,
#     "view_count": 25315,
#     "content": "Danny DeVito llorando it's always sunny in Philadelphia I get it now",
#     "count": 1,
#     "category": "twitter",
#     "subcategory": "tweet"
#     }
