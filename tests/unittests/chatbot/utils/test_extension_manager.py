# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Tests for extension_manager.py functionality."""

from __future__ import annotations

import asyncio
import datetime

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List, cast

import discord
import pytest_structlog
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


@pytest.fixture
def extension_test_configure(log: pytest_structlog.StructuredLogCapture) -> None:
    """Configure structlog for extension loading tests.

    Args:
        log: The LogCapture fixture
    """
    # Configure pytest-structlog to keep all processors
    log.keep_all_processors = True

    structlog.configure(
        processors=[
            # Add stdlib processors first
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            # Add structlog processors
            structlog.contextvars.merge_contextvars,  # Add this to capture bound context
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,  # Ensure this comes before LogCapture
            structlog.processors.UnicodeDecoder(),
            log,  # LogCapture must be last
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,  # Important for test isolation
        context_class=dict,
    )


@pytest.mark.skip_until(
    deadline=datetime.datetime(2026, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
@pytest.mark.asyncio
async def test_load_extension_with_retry_logging(
    bot: DemocracyBot,
    mocker: MockerFixture,
    extension_test_configure: None,
    log: pytest_structlog.StructuredLogCapture,
) -> None:
    """Test extension loading log messages.

    This test verifies that appropriate log messages are generated during extension loading,
    including retry attempts and final success/failure messages. It checks:
    1. Retry attempt logs with proper levels and exception details
    2. Final success log with proper level
    3. Log message order
    4. Extension name presence in all logs
    5. Proper timestamp formatting
    6. Exception details in retry logs
    7. Bound context variables

    Args:
        bot: The bot instance to test
        mocker: Pytest mocker fixture
        extension_test_configure: Fixture to configure structlog
        log: The LogCapture fixture
    """
    # Create specific exceptions for testing
    first_error = Exception("First try")
    second_error = Exception("Second try")

    # Mock load_extension to fail twice then succeed
    mock_load = mocker.patch.object(bot, "load_extension")
    mock_load.side_effect = [first_error, second_error, None]

    with structlog.testing.capture_logs() as captured:
        await load_extension_with_retry(bot, "test_extension", 3)

        # Verify retry attempt logs
        retry_logs = [
            log
            for log in captured
            if log.get("event") == "Extension load attempt failed, retrying" and log.get("log_level") == "warning"
        ]
        assert len(retry_logs) == 2, "Expected exactly two retry warning logs"

        # Verify first retry log
        first_retry = retry_logs[0]
        assert first_retry.get("operation") == "load_extension_retry", "Missing operation context"
        assert first_retry.get("extension") == "test_extension", "Missing extension context"
        assert first_retry.get("attempt") == 1, "Missing or incorrect attempt number"
        assert first_retry.get("error") == "First try", "Missing or incorrect error message"
        assert first_retry.get("error_type") == "Exception", "Missing or incorrect error type"
        assert first_retry.get("remaining_attempts") == 2, "Missing or incorrect remaining attempts"

        # Verify second retry log
        second_retry = retry_logs[1]
        assert second_retry.get("operation") == "load_extension_retry", "Missing operation context"
        assert second_retry.get("extension") == "test_extension", "Missing extension context"
        assert second_retry.get("attempt") == 2, "Missing or incorrect attempt number"
        assert second_retry.get("error") == "Second try", "Missing or incorrect error message"
        assert second_retry.get("error_type") == "Exception", "Missing or incorrect error type"
        assert second_retry.get("remaining_attempts") == 1, "Missing or incorrect remaining attempts"

        # Verify final success log
        success_logs = [
            log
            for log in captured
            if log.get("event") == "Extension loaded successfully" and log.get("log_level") == "info"
        ]
        assert len(success_logs) == 1, "Expected exactly one success log"
        success_log = success_logs[0]
        assert success_log.get("operation") == "load_extension_retry", "Missing operation context"
        assert success_log.get("extension") == "test_extension", "Missing extension context"

        # Verify log order - retries should come before success
        events = [log.get("event") for log in captured]
        first_retry_idx = events.index("Extension load attempt failed, retrying")
        second_retry_idx = events.index("Extension load attempt failed, retrying", first_retry_idx + 1)
        success_idx = events.index("Extension loaded successfully")

        assert first_retry_idx < second_retry_idx < success_idx, "Log messages not in expected order"

        # Verify all logs have required fields
        for log_entry in captured:
            assert "timestamp" in log_entry, "Missing timestamp"
            assert "logger" in log_entry, "Missing logger name"
            assert log_entry.get("log_level") in ["warning", "info"], "Invalid log level"
            assert log_entry.get("operation") == "load_extension_retry", "Missing operation context"
            assert log_entry.get("extension") == "test_extension", "Missing extension context"

        # Verify total number of logs
        assert len(captured) == 3, "Expected exactly three log messages"
