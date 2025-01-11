"""Tests for extension_manager.py functionality."""

from __future__ import annotations

import asyncio

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List, cast

import discord
import structlog

from discord.ext import commands

import pytest

from democracy_exe.chatbot.core.bot import DemocracyBot
from democracy_exe.chatbot.utils.extension_manager import get_extension_load_order, load_extension_with_retry


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


def test_get_extension_load_order() -> None:
    """Test extension load order resolution."""
    extensions = [
        "democracy_exe.chatbot.cogs.admin",
        "democracy_exe.chatbot.cogs.core",
        "democracy_exe.chatbot.cogs.ai",
        "democracy_exe.chatbot.cogs.utils",
    ]

    order = get_extension_load_order(extensions)

    # Core should be first since others depend on it
    assert order[0] == "democracy_exe.chatbot.cogs.core"

    # Other extensions should come after core
    for ext in order[1:]:
        assert ext in [
            "democracy_exe.chatbot.cogs.admin",
            "democracy_exe.chatbot.cogs.ai",
            "democracy_exe.chatbot.cogs.utils",
        ]


def test_get_extension_load_order_circular_dependency() -> None:
    """Test circular dependency detection."""
    # Create a circular dependency by adding core as dependent on admin
    extensions = ["democracy_exe.chatbot.cogs.admin", "democracy_exe.chatbot.cogs.core"]

    # Create a circular dependency by making core depend on admin and admin depend on core
    dependencies = {
        "democracy_exe.chatbot.cogs.admin": {"democracy_exe.chatbot.cogs.core"},
        "democracy_exe.chatbot.cogs.core": {"democracy_exe.chatbot.cogs.admin"},
    }

    with pytest.raises(ValueError, match="Circular dependency detected"):
        get_extension_load_order(extensions, dependencies)


@pytest.mark.asyncio
async def test_load_extension_with_retry_success(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test successful extension loading.

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Mock load_extension to succeed
    mock_load = mocker.patch.object(bot, "load_extension")
    mock_load.return_value = None

    await load_extension_with_retry(bot, "test_extension", 3)

    # Should succeed on first try
    mock_load.assert_called_once_with("test_extension")


@pytest.mark.asyncio
async def test_load_extension_with_retry_failure(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test extension loading failure after retries.

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Mock load_extension to always fail
    mock_load = mocker.patch.object(bot, "load_extension")
    mock_load.side_effect = Exception("Load failed")

    with pytest.raises(RuntimeError, match="Failed to load extension .* after 3 attempts"):
        await load_extension_with_retry(bot, "test_extension", 3)

    # Should have tried 3 times
    assert mock_load.call_count == 3


@pytest.mark.asyncio
async def test_load_extension_with_retry_eventual_success(bot: DemocracyBot, mocker: MockerFixture) -> None:
    """Test extension loading succeeding after retries.

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    # Mock load_extension to fail twice then succeed
    mock_load = mocker.patch.object(bot, "load_extension")
    mock_load.side_effect = [Exception("First try"), Exception("Second try"), None]

    await load_extension_with_retry(bot, "test_extension", 3)

    # Should have tried 3 times
    assert mock_load.call_count == 3


@pytest.mark.asyncio
async def test_load_extension_with_retry_logging(
    bot: DemocracyBot, mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    """Test extension loading log messages.

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
        caplog: Pytest log capture fixture
    """
    # Mock load_extension to fail once then succeed
    mock_load = mocker.patch.object(bot, "load_extension")
    mock_load.side_effect = [Exception("First try"), None]

    with structlog.testing.capture_logs() as captured:
        await load_extension_with_retry(bot, "test_extension", 3)

        # Should have warning for failure and info for success
        assert any(
            log.get("event") == "Failed to load extension test_extension (attempt 1/3): First try" for log in captured
        ), "Missing retry warning log"
        assert any(log.get("event") == "Loaded extension: test_extension" for log in captured), "Missing success log"
