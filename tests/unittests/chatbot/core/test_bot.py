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

from democracy_exe.chatbot.core.bot import DemocracyBot
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
async def test_bot_task_limit(bot: DemocracyBot) -> None:
    """Test bot task limit enforcement.

    Args:
        bot: The bot instance to test
    """
    # Create tasks up to limit
    for _ in range(bot.MAX_CONCURRENT_TASKS):
        task = asyncio.create_task(asyncio.sleep(0.1))
        bot.tasks.append(task)

    # Adding one more should raise
    with pytest.raises(RuntimeError, match="Too many concurrent tasks"):
        task = asyncio.create_task(asyncio.sleep(0.1))
        bot.tasks.append(task)

    # Cleanup tasks
    for task in list(bot.tasks):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_bot_setup_hook(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test bot setup hook.

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Mock extension loading
    mock_load = mocker.patch("democracy_exe.chatbot.utils.extension_utils.load_extension_with_retry")

    # Run setup hook
    await bot._async_setup_hook()

    # Should attempt to load extensions
    assert mock_load.call_count > 0


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
