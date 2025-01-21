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
import structlog

from discord.ext import commands

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


@pytest.mark.asyncio
async def test_bot_setup_timeout(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test bot setup timeout handling.

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """

    # Mock extension loading to take too long
    async def slow_load(*args: Any, **kwargs: Any) -> None:
        await asyncio.sleep(2)

    mocker.patch("democracy_exe.chatbot.utils.extension_utils.load_extension_with_retry", side_effect=slow_load)

    # Setup should timeout
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(1):
            await bot._async_setup_hook()


@pytest.mark.asyncio
async def test_bot_command_error_handling(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test bot command error handling.

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Create a mock context
    ctx = mocker.Mock()
    ctx.send = mocker.AsyncMock()

    # Test various error types
    errors = [
        commands.MissingPermissions(["manage_messages"]),
        commands.BotMissingPermissions(["manage_messages"]),
        commands.MissingRole("Admin"),
        commands.NSFWChannelRequired(ctx.channel),
        commands.NoPrivateMessage(),
        commands.MissingRequiredArgument(mocker.Mock()),
        ValueError("Custom error"),
    ]

    for error in errors:
        await bot.on_command_error(ctx, error)
        ctx.send.assert_called()
        ctx.send.reset_mock()


@pytest.mark.asyncio
async def test_bot_message_size_limit(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test bot message size limit handling.

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Create a message that's too large
    large_content = "x" * (bot.MAX_MESSAGE_LENGTH + 1)

    # Sending should raise
    with pytest.raises(ValueError, match="Message too long"):
        await bot.send_message(mocker.Mock(), large_content)


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
