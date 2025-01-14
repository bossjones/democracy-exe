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
"""Tests for the DemocracyBot class.

This module contains tests for verifying the functionality of the DemocracyBot,
particularly focusing on resource management and error handling.
"""

from __future__ import annotations

import asyncio
import datetime

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import discord
import structlog

from discord.ext import commands

import pytest

from pytest_mock import MockerFixture

from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.core.bot import DemocracyBot
from democracy_exe.chatbot.utils.resource_manager import ResourceLimits, ResourceManager


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def test_settings(monkeypatch: MonkeyPatch) -> None:
    """Configure test settings.

    Args:
        monkeypatch: The pytest monkeypatch fixture
    """
    monkeypatch.setattr(aiosettings, "discord_client_id", "test_client_id")
    monkeypatch.setattr(aiosettings, "discord_client_secret", "test_client_secret")
    monkeypatch.setattr(aiosettings, "discord_token", "test_token")


@pytest.fixture
async def bot(test_settings: None) -> AsyncGenerator[DemocracyBot, None]:
    """Create a DemocracyBot instance for testing.

    Args:
        test_settings: The test settings fixture

    Yields:
        DemocracyBot: The bot instance for testing
    """
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.members = True

    bot = DemocracyBot(command_prefix="?", intents=intents)
    # Mock the bot's user for ID checks
    bot.user = discord.Object(id=123456789)
    yield bot
    await bot.cleanup()


@pytest.fixture
def mock_message(mocker: MockerFixture) -> discord.Message:
    """Create a mock Discord message.

    Args:
        mocker: The pytest mocker fixture

    Returns:
        discord.Message: A mock message instance
    """
    message = mocker.Mock(spec=discord.Message)
    message.content = "Test message"
    message.attachments = []
    # Mock the author with an ID
    message.author = discord.Object(id=987654321)
    return message


@pytest.mark.asyncio
async def test_bot_initialization(bot: DemocracyBot) -> None:
    """Test that the bot initializes with proper resource limits.

    Args:
        bot: The bot instance
    """
    assert isinstance(bot.resource_manager, ResourceManager)
    assert bot.resource_manager.limits.max_memory_mb == 512
    assert bot.resource_manager.limits.max_tasks == 100
    assert bot.resource_manager.limits.max_response_size_mb == 8
    assert bot.resource_manager.limits.max_buffer_size_kb == 64
    assert bot.resource_manager.limits.task_timeout_seconds == 300


@pytest.mark.asyncio
async def test_get_context_memory_check(
    bot: DemocracyBot, mock_message: discord.Message, mocker: MockerFixture
) -> None:
    """Test that get_context performs memory checks.

    Args:
        bot: The bot instance
        mock_message: A mock message
        mocker: The pytest mocker fixture
    """
    mock_check = mocker.patch.object(bot.resource_manager, "check_memory")
    await bot.get_context(mock_message)
    mock_check.assert_called_once()


@pytest.mark.asyncio
async def test_get_context_message_size_limit(bot: DemocracyBot, mock_message: discord.Message) -> None:
    """Test that get_context enforces message size limits.

    Args:
        bot: The bot instance
        mock_message: A mock message
    """
    # Create large message content
    mock_message.content = "x" * (bot.resource_manager.limits.max_response_size_mb * 1024 * 1024 + 1)

    with pytest.raises(RuntimeError, match="Message size .* exceeds limit"):
        await bot.get_context(mock_message)


@pytest.mark.asyncio
async def test_get_context_attachment_size_limit(
    bot: DemocracyBot, mock_message: discord.Message, mocker: MockerFixture
) -> None:
    """Test that get_context enforces attachment size limits.

    Args:
        bot: The bot instance
        mock_message: A mock message
        mocker: The pytest mocker fixture
    """
    # Create large attachment
    mock_attachment = mocker.Mock(spec=discord.Attachment)
    mock_attachment.size = bot.resource_manager.limits.max_response_size_mb * 1024 * 1024 + 1
    mock_message.attachments = [mock_attachment]

    with pytest.raises(RuntimeError, match="Total attachment size .* exceeds limit"):
        await bot.get_context(mock_message)


@pytest.mark.asyncio
async def test_add_task_tracking(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test that add_task properly tracks tasks.

    Args:
        bot: The bot instance
        mocker: The pytest mocker fixture
    """
    mock_track = mocker.patch.object(bot.resource_manager, "track_task")
    mock_cleanup = mocker.patch.object(bot.resource_manager, "cleanup_tasks")

    async def test_coro() -> None:
        await asyncio.sleep(0.1)

    await bot.add_task(test_coro())
    mock_track.assert_called_once()
    await asyncio.sleep(0.2)  # Wait for task to complete
    mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_add_task_timeout(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test that add_task enforces timeouts.

    Args:
        bot: The bot instance
        mocker: The pytest mocker fixture
    """
    mock_cleanup = mocker.patch.object(bot.resource_manager, "cleanup_tasks")
    bot.resource_manager.limits.task_timeout_seconds = 0.1

    async def long_task() -> None:
        await asyncio.sleep(1.0)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(bot.add_task(long_task()), timeout=0.2)

    await asyncio.sleep(0.2)  # Wait for cleanup
    mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test that cleanup properly releases resources.

    Args:
        bot: The bot instance
        mocker: The pytest mocker fixture
    """
    mock_force_cleanup = mocker.patch.object(bot.resource_manager, "force_cleanup")
    await bot.cleanup()
    mock_force_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_gateway_data_cleanup(bot: DemocracyBot) -> None:
    """Test that gateway data is properly cleaned up.

    Args:
        bot: The bot instance
    """
    # Add old data
    one_week_ago = discord.utils.utcnow() - datetime.timedelta(days=8)
    recent_date = discord.utils.utcnow()

    bot.identifies[0] = [one_week_ago, recent_date]
    bot.resumes[0] = [one_week_ago, recent_date]

    bot._clear_gateway_data()

    # Check that old data was removed but recent data remains
    assert len(bot.identifies[0]) == 1
    assert len(bot.resumes[0]) == 1
    assert bot.identifies[0][0] == recent_date
    assert bot.resumes[0][0] == recent_date
