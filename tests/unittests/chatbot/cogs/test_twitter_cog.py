# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
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
import rich
import structlog

from discord.client import _LoopSentinel
from discord.ext import commands
from structlog.testing import capture_logs

import pytest

from democracy_exe.chatbot.cogs.twitter import HELP_MESSAGE, TwitterError
from democracy_exe.chatbot.cogs.twitter import Twitter as TwitterCog
from democracy_exe.chatbot.core.bot import DemocracyBot
from democracy_exe.utils.twitter_utils.embed import create_error_embed, create_info_embed
from democracy_exe.utils.twitter_utils.models import TweetInfo
from democracy_exe.utils.twitter_utils.types import TweetDownloadMode, TweetMetadata
from tests.tests_utils.last_ctx_cog import LastCtxCog


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

    from democracy_exe.chatbot.core.bot import DemocracyBot

logger = structlog.get_logger(__name__)

# Constants for testing
TEST_TWEET_URL = "https://x.com/bancodevideo/status/1699925133194858974"


@pytest_asyncio.fixture
async def mock_tweet_info_data() -> AsyncGenerator[dict[str, Any], None]:
    """Load mock tweet info data from fixture file.

    Yields:
        dict[str, Any]: Mock tweet info data
    """
    async with aiofiles.open("tests/fixtures/info.json", encoding="utf-8") as f:
        content = await f.read()
        data = json.loads(content)
        yield data


@pytest_asyncio.fixture
async def mock_tweet_media_data() -> AsyncGenerator[dict[str, Any], None]:
    """Load mock tweet media data from fixture file.

    Yields:
        dict[str, Any]: Mock tweet media data
    """
    async with aiofiles.open(
        "tests/fixtures/Eminitybaba_-1868256259251863704-(20241215_112617)-img1.mp4.json", encoding="utf-8"
    ) as f:
        content = await f.read()
        data = json.loads(content)
        yield data


@pytest_asyncio.fixture
async def mock_tweet_metadata(mock_tweet_info_data: dict[str, Any]) -> AsyncGenerator[TweetMetadata, None]:
    """Convert mock tweet info data into TweetMetadata object.

    Args:
        mock_tweet_info_data: Mock tweet info data from fixture

    Yields:
        TweetMetadata: Converted tweet metadata object
    """
    data_model = TweetInfo(**mock_tweet_info_data)
    metadata = TweetMetadata(
        id=str(data_model.tweet_id),
        url=TEST_TWEET_URL,  # Using the test URL constant
        author=data_model.author.name,  # pylint: disable=no-member
        content=data_model.content,
        media_urls=[],  # This would typically come from _get_media_urls
        created_at=data_model.date,
    )
    yield metadata


@pytest_asyncio.fixture(autouse=True, scope="function")
async def bot_with_twitter_cog() -> AsyncGenerator[DemocracyBot, None]:
    # async def bot_with_twitter_cog() -> DemocracyBot:
    """Create a DemocracyBot instance for testing.

    Args:
        event_loop: The event loop fixture

    Returns:
        AsyncGenerator[DemocracyBot, None]: DemocracyBot instance with test configuration
    """
    # Configure intents
    # intents = discord.Intents.default()
    intents = discord.Intents.all()
    # intents.members = True
    # intents.message_content = True
    # intents.messages = True
    # intents.guilds = True

    intents.message_content = True
    intents.guilds = True
    intents.members = True
    intents.bans = True
    intents.emojis = True
    intents.voice_states = True
    intents.messages = True
    intents.reactions = True

    test_bot = DemocracyBot(command_prefix="?", intents=intents, description="Test DemocracyBot instance")
    rich.print("<green>BEFORE</green>")
    # rich.inspect(test_bot, all=True)

    # import bpdb; bpdb.set_trace()

    # set up the loop
    # if isinstance(test_bot.loop, _LoopSentinel):  # type: ignore
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

    rich.print("<green>AFTER</green>")
    # rich.inspect(test_bot, all=True)

    # import bpdb; bpdb.set_trace()

    # return test_bot
    yield test_bot

    # # await dpytest.empty_queue()

    try:
        # Teardown
        await dpytest.empty_queue()  # empty the global message queue as test teardown
    finally:
        pass


@pytest_asyncio.fixture(autouse=True, scope="function")
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


@pytest_asyncio.fixture
async def mock_tweet_data_with_media(mock_tweet_data: dict[str, Any], tmp_path: pathlib.Path) -> dict[str, Any]:
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
    async with aiofiles.open(test_file, mode="w", encoding="utf-8") as f:
        await f.write("test content")  # type: ignore[arg-type]

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
    # media_path.mkdir(parents=True, exist_ok=True)

    # Create test media files
    test_video = media_path / "test_video.mp4"
    test_image = media_path / "test_image.jpg"

    # Write some dummy content to the files
    async with aiofiles.open(test_video, mode="wb") as f:
        await f.write(b"dummy video content")  # type: ignore[arg-type]
    async with aiofiles.open(test_image, mode="wb") as f:
        await f.write(b"dummy image content")  # type: ignore[arg-type]

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
    async with aiofiles.open(info_json, mode="w", encoding="utf-8") as f:
        await f.write(json.dumps(info_data))  # type: ignore[arg-type]

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
    yield data, tmp_path
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

    # NOTE: temporarliy disabling
    # try:
    #     for file in [test_video, test_image, info_json]:
    #         if file.exists():
    #             file.unlink()
    #     media_path.rmdir()
    #     # await aiofiles.os.rmdir(f"{media_path}")
    # except Exception as e:
    #     logger.warning(f"Cleanup error in mock_aio_tweet_data_with_media: {e}")


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
@pytest.mark.asyncio
async def test_twitter_cog_init(bot_with_twitter_cog: DemocracyBot) -> None:
    """Test Twitter cog initialization.

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
    """
    cog = bot_with_twitter_cog.get_cog("Twitter")
    assert isinstance(cog, TwitterCog)
    assert isinstance(cog.bot, commands.Bot)


@pytest.mark.asyncio
async def test_twitter_cog_on_ready(bot_with_twitter_cog: DemocracyBot, caplog: LogCaptureFixture) -> None:
    """Test Twitter cog on_ready event.

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        cog = bot_with_twitter_cog.get_cog("Twitter")
        await cog.on_ready()  # type: ignore

        # Check if the log message exists in the captured structlog events
        assert any(log.get("event") == "Twitter Cog ready." for log in captured), (
            "Expected 'Twitter Cog ready.' message not found in logs"
        )


@pytest.mark.asyncio
async def test_twitter_cog_on_guild_join(bot_with_twitter_cog: DemocracyBot, caplog: LogCaptureFixture) -> None:
    """Test Twitter cog on_guild_join event.

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        cog = bot_with_twitter_cog.get_cog("Twitter")
        guild = bot_with_twitter_cog.guilds[0]
        await cog.on_guild_join(guild)  # type: ignore

        # Check if the log message exists in the captured structlog events
        assert any(log.get("event") == f"Adding new guild to database: {guild.id}" for log in captured), (
            f"Expected 'Adding new guild to database: {guild.id}' message not found in logs"
        )


# NOTE: to get this to to pass after the refactor, you might need to incorporate things like which channel this is being said in etc. (via aider)
#  # Create channel with specific ID
#  channel = await dpytest.driver.create_text_channel(
#      guild,
#      channel_id=1240294186201124929
#  )

#  # Send message in that specific channel
#  await dpytest.message("?tweet", channel=channel)


@pytest.mark.asyncio
async def test_tweet_help_command(bot_with_twitter_cog: DemocracyBot) -> None:
    """Test tweet help command shows help message.

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
    """
    with capture_logs() as captured:
        await dpytest.message("?tweet")
        # assert dpytest.verify().message().content(HELP_MESSAGE)

        # Verify expected log events
        assert any(log.get("event") == "AI is disabled, skipping message processing... with llm" for log in captured), (
            "Expected 'AI is disabled' message not found in logs"
        )

        assert any(log.get("event") == "Tweet command invoked by TestUser0#0001 in Test Guild 0" for log in captured), (
            "Expected tweet command invocation message not found in logs"
        )

        assert any(log.get("event") == "No subcommand specified, sending help message" for log in captured), (
            "Expected help message event not found in logs"
        )


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Still figuring out how to tech llm's to test with dpytest",
)
@pytest.mark.asyncio
async def test_download_tweet_failure(
    bot_with_twitter_cog: DemocracyBot,
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
) -> None:
    """Test tweet download failure handling.

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
        mocker: Pytest mocker fixture
        caplog: Pytest log capture fixture
        capsys: Pytest stdout/stderr capture fixture
    """
    with capsys.disabled():
        with capture_logs() as captured:
            # Mock shell command execution
            mock_shell = mocker.AsyncMock(return_value=(b"", b""))
            mocker.patch("democracy_exe.shell._aio_run_process_and_communicate", side_effect=mock_shell)

            error_msg = "Failed to download tweet"
            # Mock download_tweet with a side effect to print when it's called
            mock_result = {"success": False, "error": error_msg, "metadata": {}, "local_files": []}

            async def mock_download_side_effect(*args, **kwargs):
                print("Mock download_tweet called!")
                return mock_result

            mock_download = mocker.patch(
                "democracy_exe.utils.twitter_utils.download.download_tweet",
                side_effect=mock_download_side_effect,
                new_callable=mocker.AsyncMock,  # Add this line to make it async
            )

            await dpytest.message(f"?tweet download {TEST_TWEET_URL}")

            dummy_embed = discord.Embed(
                title="Download in Progress",
                description=f"Downloading tweet single from {TEST_TWEET_URL}...",
                color=discord.Color.gold(),
            )

            # Wait for messages to be processed
            await asyncio.sleep(0.1)

            try:
                # First message should be progress embed
                assert dpytest.verify().message().peek().embed(dummy_embed)

                mock_download.assert_called_once()
                mock_shell.assert_called()

                # Verify logs using structlog's capture_logs
                assert any(log.get("event") == "Download command invoked" for log in captured), (
                    "Expected 'Download command invoked' message not found in logs"
                )

                assert any(log.get("event") == "Download failed" for log in captured), (
                    "Expected 'Download failed' message not found in logs"
                )

                assert any(log.get("event") == error_msg for log in captured), (
                    f"Expected '{error_msg}' message not found in logs"
                )
            finally:
                # Cleanup
                await dpytest.empty_queue()


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Still figuring out how to tech llm's to test with dpytest",
)
@pytest.mark.asyncio
async def test_thread_command(
    bot_with_twitter_cog: DemocracyBot, mock_tweet_data: dict[str, Any], mocker: MockerFixture
) -> None:
    """Test thread download command.

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
        mock_tweet_data: Mock tweet data fixture
        mocker: Pytest mocker fixture
    """
    # Mock shell command execution
    mock_shell = mocker.AsyncMock(return_value=(b"", b""))
    mocker.patch("democracy_exe.shell._aio_run_process_and_communicate", side_effect=mock_shell)

    mock_download = mocker.patch(
        "democracy_exe.utils.twitter_utils.download.download_tweet", return_value=mock_tweet_data
    )

    await dpytest.message(f"?tweet thread {TEST_TWEET_URL}")
    # Get progress message
    progress = dpytest.get_message()
    # Get completion message
    completion = dpytest.get_message()
    assert completion.content == "Download complete!"
    mock_download.assert_called_once_with(TEST_TWEET_URL, mode="thread")
    mock_shell.assert_called()


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Still figuring out how to tech llm's to test with dpytest",
)
@pytest.mark.asyncio
async def test_card_command(
    bot_with_twitter_cog: DemocracyBot, mock_tweet_data: dict[str, Any], mocker: MockerFixture
) -> None:
    """Test card download command.

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
        mock_tweet_data: Mock tweet data fixture
        mocker: Pytest mocker fixture
    """
    # Mock shell command execution
    mock_shell = mocker.AsyncMock(return_value=(b"", b""))
    mocker.patch("democracy_exe.shell._aio_run_process_and_communicate", side_effect=mock_shell)

    mock_download = mocker.patch(
        "democracy_exe.utils.twitter_utils.download.download_tweet", return_value=mock_tweet_data
    )

    await dpytest.message(f"?tweet card {TEST_TWEET_URL}")
    # Get progress message
    progress = dpytest.get_message()
    # Get completion message
    completion = dpytest.get_message()
    assert completion.content == "Download complete!"
    mock_download.assert_called_once_with(TEST_TWEET_URL, mode="card")
    mock_shell.assert_called()


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Still figuring out how to tech llm's to test with dpytest",
)
@pytest.mark.asyncio
async def test_info_command(
    bot_with_twitter_cog: DemocracyBot, mock_tweet_data: dict[str, Any], mocker: MockerFixture
) -> None:
    """Test tweet info command.

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
        mock_tweet_data: Mock tweet data fixture
        mocker: Pytest mocker fixture
    """
    # Mock shell command execution
    mock_shell = mocker.AsyncMock(return_value=(b"", b""))
    mocker.patch("democracy_exe.shell._aio_run_process_and_communicate", side_effect=mock_shell)

    mock_download = mocker.patch(
        "democracy_exe.utils.twitter_utils.download.download_tweet", return_value=mock_tweet_data
    )

    await dpytest.message(f"?tweet info {TEST_TWEET_URL}")
    messages = dpytest.get_message()
    expected_embed = create_info_embed(mock_tweet_data["metadata"])
    actual_embed = messages.embeds[0]  # type: ignore

    # Compare relevant fields
    assert actual_embed.title == expected_embed.title
    assert actual_embed.description == expected_embed.description
    assert actual_embed.color == expected_embed.color

    # Compare field values
    actual_fields = {f.name: f.value for f in actual_embed.fields}
    expected_fields = {f.name: f.value for f in expected_embed.fields}
    assert actual_fields == expected_fields
    mock_create_info_embed = mocker.patch("democracy_exe.utils.twitter_utils.embed.create_info_embed")
    mock_create_info_embed.return_value = expected_embed
    mock_create_info_embed.assert_called_once_with(mock_tweet_data["metadata"])
    mock_download.assert_called_once()
    mock_shell.assert_called()


@pytest.mark.asyncio
async def test_cleanup_temp_dir(
    bot_with_twitter_cog: DemocracyBot, tmp_path: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """Test temporary directory cleanup.

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
        tmp_path: Pytest temporary path fixture
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        cog = bot_with_twitter_cog.get_cog("Twitter")
        test_dir = tmp_path / "gallery-dl" / "test"
        test_dir.mkdir(parents=True)
        test_file = test_dir / "test.txt"
        test_file.touch()

        cog._cleanup_temp_dir(str(test_file))  # type: ignore
        assert not test_dir.exists()

        # Verify cleanup was logged using structlog's capture_logs
        assert any(log.get("event") == "Temporary directory cleanup complete" for log in captured), (
            "Expected 'Temporary directory cleanup complete' message not found in logs"
        )


async def test_download_tweet_success_twitter_cog(
    bot_with_twitter_cog: DemocracyBot,
    mocker: MockerFixture,
    mock_aio_tweet_data_with_media: tuple[dict[str, Any], pathlib.Path],
    mock_tweet_metadata: TweetMetadata,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
) -> None:
    """Test successful tweet download functionality.

    This test verifies that the tweet download command works correctly by:
    1. Mocking shell command execution
    2. Mocking tweet download functionality
    3. Verifying proper message responses
    4. Checking command completion

    Args:
        bot_with_twitter_cog: The Discord bot instance with Twitter cog
        mocker: Pytest mocker for patching functions
        mock_aio_tweet_data_with_media: Mock tweet data with media files
        caplog: Fixture for capturing log output
        capsys: Fixture for capturing system output

    Raises:
        AssertionError: If expected messages are not received or in wrong format
    """
    with capsys.disabled():
        with capture_logs() as captured:
            caplog.set_level(logging.DEBUG)

            mock_aio_tweet_data, tmp_path = mock_aio_tweet_data_with_media

            # ********************************************* (testing this out) ********************************

            media_path = tmp_path / "gallery-dl" / "twitter" / "test_user"

            await aiofiles.os.makedirs(f"{media_path}", exist_ok=True)
            # bpdb.set_trace()
            # media_path.mkdir(parents=True, exist_ok=True)

            # Create test media files
            test_video = media_path / "test_video.mp4"
            test_image = media_path / "test_image.jpg"

            # Write some dummy content to the files
            async with aiofiles.open(test_video, mode="wb") as f:
                await f.write(b"dummy video content")  # type: ignore[arg-type]
            async with aiofiles.open(test_image, mode="wb") as f:
                await f.write(b"dummy image content")  # type: ignore[arg-type]

            # Create info.json file
            info_json = media_path / "info.json"
            # info_data = {
            #     "tweet": {
            #         "id": "123456789",
            #         "url": TEST_TWEET_URL,
            #         "text": "Test tweet with media content",
            #         "created_at": "2024-01-01",
            #         "media": [{"url": "https://test.com/video.mp4"}, {"url": "https://test.com/image.jpg"}],
            #         "user": {"name": "test_user", "screen_name": "test_user"},
            #     }
            # }
            info_data = mock_tweet_metadata
            async with aiofiles.open(info_json, mode="w", encoding="utf-8") as f:
                await f.write(json.dumps(info_data))  # type: ignore[arg-type]

            # ********************************************* (testing this out) ********************************
            # # Create mock data dictionary
            # data = {
            #     "success": True,
            #     "metadata": {
            #         "id": "123456789",
            #         "url": TEST_TWEET_URL,
            #         "author": "test_user",
            #         "content": "Test tweet content",
            #         "media_urls": [],
            #         "created_at": "2024-01-01",
            #     },
            #     "local_files": [],
            #     "error": None,
            # }
            # data.update({
            #     "success": True,
            #     "metadata": {
            #         "id": "123456789",
            #         "url": TEST_TWEET_URL,
            #         "author": "test_user",
            #         "content": "Test tweet with media content",
            #         "media_urls": ["https://test.com/video.mp4", "https://test.com/image.jpg"],
            #         "created_at": "2024-01-01",
            #     },
            #     "local_files": [str(test_video), str(test_image)],
            #     "error": None,
            # })
            data = mock_tweet_metadata

            # ********************************************* (testing this out) ********************************

            # Mock shell command execution with AsyncMock
            mock_shell = mocker.AsyncMock(return_value=(b"", b""))
            mocker.patch("democracy_exe.shell._aio_run_process_and_communicate", side_effect=mock_shell)

            # # Mock shell command execution
            # mock_shell = mocker.AsyncMock()
            # mock_shell.return_value = (0, "Success", "")  # Return code 0, stdout, stderr
            # mocker.patch("democracy_exe.utils.shell._aio_run_process_and_communicate", mock_shell)

            from democracy_exe.utils.twitter_utils.types import DownloadResult

            mock_download_result: DownloadResult = DownloadResult(**mock_aio_tweet_data)
            # Mock download_tweet function
            mock_download = mocker.patch("democracy_exe.utils.twitter_utils.download.download_tweet")
            mock_download.return_value = mock_download_result
            test_media = mock_aio_tweet_data["local_files"][0]

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
            await dpytest.message(f"?tweet download {TEST_TWEET_URL}")

            # Wait for messages to be processed
            await asyncio.sleep(0.1)

            # assert dpytest.verify().message().contains().content("Download complete!")
            message = dpytest.get_message(peek=True)
            assert message is not None
            assert message.content == "Download complete!"

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

            try:
                # Teardown
                await dpytest.empty_queue()  # empty the global message queue as test teardown
            finally:
                pass
