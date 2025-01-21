# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Tests for DemocracyBot class functionality."""

from __future__ import annotations

import asyncio

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List, cast

import discord
import pytest_structlog
import structlog

from discord.ext import commands
from structlog.processors import TimeStamper, add_log_level
from structlog.stdlib import BoundLogger
from structlog.testing import CapturingLogger, LogCapture
from structlog.typing import BindableLogger, EventDict, Processor, WrappedLogger

import pytest

from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.core.bot import DemocracyBot
from democracy_exe.chatbot.utils.discord_utils import extensions as discord_extensions
from democracy_exe.chatbot.utils.extension_utils import get_extension_load_order, load_extension_with_retry


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture(autouse=True)
def configure_structlog(log: pytest_structlog.StructuredLogCapture) -> None:
    """Configure structlog for testing.

    This fixture ensures structlog is properly configured for capturing logs in tests.
    The configuration follows best practices from test_logsetup.py and uses
    pytest-structlog plugin.

    Args:
        log: The pytest-structlog capture fixture
    """
    # Reset any previous configuration
    structlog.reset_defaults()

    structlog.configure(
        processors=[
            # Add stdlib processors first
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            # Add structlog processors
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            # LogCapture must be last
            log,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,  # Important for test isolation
    )


@pytest.fixture
async def bot() -> AsyncGenerator[DemocracyBot, None]:
    """Create a DemocracyBot instance for testing.

    Yields:
        DemocracyBot: The bot instance for testing
    """
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    bot = DemocracyBot(command_prefix="?", intents=intents)
    yield bot
    await bot.cleanup()


@pytest.mark.asyncio
async def test_bot_initialization(bot: DemocracyBot) -> None:
    """Test bot initialization.

    Args:
        bot: The bot instance to test
    """
    assert bot.command_prefix == "?"
    assert bot.intents.message_content is True
    assert bot.intents.guilds is True


@pytest.mark.asyncio
async def test_bot_cleanup(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test bot cleanup functionality.

    This test verifies that the bot properly cleans up all resources including:
    - Resource manager cleanup is called to handle tasks
    - Redis pool is disconnected if present
    - Message and attachment handlers are cleaned up
    - Garbage collection is triggered

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Mock resource manager and handlers
    mock_resource_manager = mocker.AsyncMock()
    mock_resource_manager.force_cleanup = mocker.AsyncMock()
    bot.resource_manager = mock_resource_manager

    mock_message_handler = mocker.AsyncMock()
    mock_message_handler.cleanup = mocker.AsyncMock()
    bot.message_handler = mock_message_handler

    mock_attachment_handler = mocker.AsyncMock()
    mock_attachment_handler.cleanup = mocker.AsyncMock()
    bot.attachment_handler = mock_attachment_handler

    # Create a mock task and add it to tasks list
    mock_task = mocker.Mock()
    bot.tasks.append(mock_task)

    # Mock Redis pool
    mock_pool = mocker.AsyncMock()
    mock_pool.disconnect = mocker.AsyncMock()
    bot.pool = mock_pool

    # Run cleanup with structlog capture
    with structlog.testing.capture_logs() as captured:
        await bot.cleanup()

        # Verify resource manager cleanup was called
        mock_resource_manager.force_cleanup.assert_awaited_once()

        # Verify Redis pool disconnect was called
        mock_pool.disconnect.assert_awaited_once()

        # Verify handlers were cleaned up
        mock_message_handler.cleanup.assert_awaited_once()
        mock_attachment_handler.cleanup.assert_awaited_once()

        # Verify cleanup was logged
        assert any(log.get("event") == "Cleanup completed successfully" for log in captured), (
            "Expected cleanup completion log not found"
        )


@pytest.mark.asyncio
async def test_bot_task_limit(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test bot task limit enforcement.

    This test verifies that the bot properly enforces task limits through
    the resource manager, including:
    - Task tracking via resource manager
    - Task limit enforcement
    - Proper error handling when limit is exceeded
    - Proper cleanup of tasks

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Mock resource manager
    mock_resource_manager = mocker.AsyncMock()
    bot.resource_manager = mock_resource_manager

    # Configure resource manager to raise on task limit
    mock_resource_manager.track_task = mocker.AsyncMock()
    mock_resource_manager.track_task.side_effect = [
        None,  # First call succeeds
        RuntimeError("Too many concurrent tasks"),  # Second call fails
    ]

    # Create and track first task - should succeed
    task1 = asyncio.create_task(asyncio.sleep(0.1))
    await bot.add_task(task1)

    # Verify task was tracked
    mock_resource_manager.track_task.assert_awaited_once()

    # Try to add another task - should fail
    task2 = asyncio.create_task(asyncio.sleep(0.1))
    with pytest.raises(RuntimeError, match="Too many concurrent tasks"):
        await bot.add_task(task2)

    # Clean up tasks
    for task in [task1, task2]:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@pytest.mark.asyncio
async def test_bot_setup_hook(mocker: MockerFixture) -> None:
    """Test the bot's setup_hook method.

    This test verifies that:
    1. The bot's application info is properly fetched and owner ID is set
    2. The resource manager is initialized
    3. Extensions are discovered and loaded correctly
    4. The invite link is generated with proper permissions

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock application info
    mock_app_info = mocker.Mock()
    mock_app_info.owner.id = 123456789
    mock_app_info.id = 987654321

    # Create bot instance
    bot = DemocracyBot()
    bot.application_info = mocker.AsyncMock(return_value=mock_app_info)

    # Mock resource manager
    mock_resource_manager = mocker.AsyncMock()
    mocker.patch("democracy_exe.chatbot.core.bot.ResourceManager", return_value=mock_resource_manager)

    # Mock extension loading
    mock_load = mocker.AsyncMock()
    bot.load_extension = mock_load

    # Mock extensions function to return initial_extensions
    mock_extensions = mocker.patch("democracy_exe.chatbot.utils.discord_utils.extensions")
    mock_extensions.return_value = aiosettings.initial_extensions

    # Call setup hook
    await bot.setup_hook()

    # Verify application info was fetched and owner ID set
    assert bot.owner_id == 123456789

    # Verify resource manager was initialized
    assert bot.resource_manager is not None

    # Verify invite link was generated
    assert bot.invite is not None
    assert "987654321" in bot.invite

    # Verify extensions were loaded
    assert mock_load.call_count == len(aiosettings.initial_extensions)
    for ext in aiosettings.initial_extensions:
        mock_load.assert_any_call(ext)


# @pytest.mark.asyncio
# async def test_bot_setup_timeout(bot: DemocracyBot, mocker: MockerFixture, log: pytest_structlog.StructuredLogCapture) -> None:
#     """Test that the bot properly handles setup timeouts.

#     This test verifies that:
#     1. The bot times out when setup takes too long
#     2. Proper error logs are generated
#     3. Resources are cleaned up
#     4. Bot enters error state

#     Args:
#         bot: The bot instance to test
#         mocker: Pytest mocker fixture
#         log: The LogCapture fixture
#     """
#     # Mock application info
#     mock_owner = mocker.AsyncMock()
#     mock_owner.id = 987654321
#     mock_owner.username = "test_owner"
#     mock_owner.discriminator = "1234"
#     mock_owner.global_name = "Test Owner"
#     mock_owner.avatar = None
#     mock_owner.bot = False
#     mock_owner.system = False
#     mock_owner.mfa_enabled = True
#     mock_owner.banner = None
#     mock_owner.accent_color = None
#     mock_owner.locale = "en-US"
#     mock_owner.verified = True
#     mock_owner.email = None
#     mock_owner.flags = 0
#     mock_owner.premium_type = 0
#     mock_owner.public_flags = 0

#     mock_app_info = mocker.AsyncMock()
#     mock_app_info.id = 123456789
#     mock_app_info.name = "Test Bot"
#     mock_app_info.description = "A test bot"
#     mock_app_info.icon = None
#     mock_app_info.rpc_origins = []
#     mock_app_info.owner = mock_owner
#     mock_app_info.verify_key = "test_key"
#     mock_app_info.team = None
#     mock_app_info.guild_id = None
#     mock_app_info.primary_sku_id = None
#     mock_app_info.slug = None
#     mock_app_info.cover_image = None
#     mock_app_info.flags = 0
#     mock_app_info.approximate_guild_count = 0
#     mock_app_info.redirect_uris = []
#     mock_app_info.interactions_endpoint_url = None
#     mock_app_info.role_connections_verification_url = None
#     mock_app_info.tags = []
#     mock_app_info.bot_public = True
#     mock_app_info.bot_require_code_grant = False
#     mock_app_info.terms_of_service_url = None
#     mock_app_info.privacy_policy_url = None

#     # Mock application_info to return our mock
#     bot.application_info = mocker.AsyncMock(return_value=mock_app_info)

#     # Mock extension loading to take longer than timeout
#     async def mock_load_extension(ext: str) -> None:
#         await asyncio.sleep(2.0)  # Longer than timeout

#     bot.load_extension = mock_load_extension

#     # Mock extensions to return a list
#     mock_extensions = mocker.patch("democracy_exe.chatbot.utils.discord_utils.extensions")
#     mock_extensions.return_value = ["democracy_exe.chatbot.cogs.test1", "democracy_exe.chatbot.cogs.test2"]

#     # Expect RuntimeError with timeout message
#     with pytest.raises(RuntimeError, match="Failed to initialize bot"):
#         await bot.start()

#     # Verify error was logged
#     assert log.has("Bot setup timed out", level="error"), "Expected timeout error log not found"

#     # Verify bot is closed
#     assert not bot.is_ready()


@pytest.mark.asyncio
async def test_bot_command_error_handling(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test bot command error handling.

    This test verifies that the bot properly handles various command errors by:
    1. Sending appropriate error messages to users
    2. Handling different error types correctly
    3. Logging errors when necessary

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Create a mock context
    ctx = mocker.Mock()
    ctx.send = mocker.AsyncMock()
    ctx.author = mocker.Mock()
    ctx.author.send = mocker.AsyncMock()
    ctx.command = mocker.Mock()
    ctx.command.qualified_name = "test_command"

    # Test various error types
    errors = [
        # Errors that send to author
        (commands.NoPrivateMessage(), "This command cannot be used in private messages.", True),
        (commands.DisabledCommand(), "Sorry. This command is disabled and cannot be used.", True),
        # Errors that send to channel
        (commands.ArgumentParsingError("Invalid argument"), "Invalid argument", False),
        # Errors that should be re-raised
        (commands.MissingPermissions(["manage_messages"]), None, False),
        (commands.BotMissingPermissions(["manage_messages"]), None, False),
        (commands.MissingRole("Admin"), None, False),
        (commands.NSFWChannelRequired(ctx.channel), None, False),
        (commands.MissingRequiredArgument(mocker.Mock()), None, False),
    ]

    # Test each error type
    for error, expected_message, send_to_author in errors:
        # Reset mock call counts
        ctx.send.reset_mock()
        ctx.author.send.reset_mock()

        # Handle the error
        if expected_message is not None:
            await bot.on_command_error(ctx, error)
            if send_to_author:
                ctx.author.send.assert_awaited_once_with(expected_message)
                assert not ctx.send.called
            else:
                ctx.send.assert_awaited_once_with(expected_message)
                assert not ctx.author.send.called
        else:
            # These errors should be re-raised
            with pytest.raises(type(error)):
                await bot.on_command_error(ctx, error)

    # Test CommandInvokeError with HTTPException
    mock_response = mocker.Mock()
    mock_response.status = 404
    http_exception = discord.HTTPException(mock_response, "Not Found")
    http_error = commands.CommandInvokeError(http_exception)
    await bot.on_command_error(ctx, http_error)
    # HTTP exceptions should not be logged
    assert not ctx.send.called
    assert not ctx.author.send.called

    # Test CommandInvokeError with other exception
    runtime_error = RuntimeError("Test error")
    other_error = commands.CommandInvokeError(runtime_error)
    await bot.on_command_error(ctx, other_error)
    # Other exceptions should be logged but not sent to user
    assert not ctx.send.called
    assert not ctx.author.send.called


@pytest.mark.asyncio
async def test_bot_message_size_limit(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test bot message size limit handling.

    This test verifies that:
    1. Messages exceeding Discord's 2000 character limit are rejected
    2. Messages within the limit are processed normally

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Mock channel for sending messages
    mock_channel = mocker.AsyncMock()

    # Test message exceeding Discord's 2000 character limit
    large_content = "x" * 2001
    with pytest.raises(ValueError, match="Message too long"):
        await bot.send_message(mock_channel, large_content)

    # Test message at exactly 2000 characters (should work)
    content_at_limit = "x" * 2000
    await bot.send_message(mock_channel, content_at_limit)
    mock_channel.send.assert_called_once_with(content_at_limit)

    # Test normal message (well under limit)
    mock_channel.reset_mock()
    normal_content = "Hello, world!"
    await bot.send_message(mock_channel, normal_content)
    mock_channel.send.assert_called_once_with(normal_content)


@pytest.mark.asyncio
async def test_bot_attachment_size_limit(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test bot attachment size limit handling.

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Create an attachment that's too large
    mock_attachment = mocker.Mock()
    mock_attachment.size = bot.MAX_ATTACHMENT_SIZE + 1

    # Processing should raise
    with pytest.raises(ValueError, match="Attachment too large"):
        await bot.process_attachment(mock_attachment)
