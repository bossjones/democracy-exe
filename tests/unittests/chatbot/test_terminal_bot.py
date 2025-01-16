"""Tests for the terminal bot module."""

from __future__ import annotations

import asyncio
import signal
import sys

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

import structlog

import pytest

from democracy_exe.chatbot.terminal_bot import ThreadSafeTerminalBot
from democracy_exe.chatbot.utils.resource_manager import ResourceLimits


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def terminal_bot() -> ThreadSafeTerminalBot:
    """Create test terminal bot.

    Returns:
        ThreadSafeTerminalBot: Test terminal bot instance
    """
    return ThreadSafeTerminalBot()


@pytest.mark.asyncio
async def test_terminal_bot_initialization(terminal_bot: ThreadSafeTerminalBot) -> None:
    """Test terminal bot initialization.

    Args:
        terminal_bot: Test terminal bot
    """
    assert terminal_bot._resource_manager is not None
    assert terminal_bot._shutdown_event is not None
    assert isinstance(terminal_bot._tasks, set)


@pytest.mark.asyncio
async def test_signal_handling(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test signal handling.

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
    """
    # Mock signal handler
    mock_set = mocker.patch.object(terminal_bot._shutdown_event, "set")

    # Simulate signal
    terminal_bot._handle_signal(signal.SIGTERM, None)

    # Verify shutdown event was set
    mock_set.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test resource cleanup.

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
    """
    # Create test task
    task = asyncio.create_task(asyncio.sleep(1))
    terminal_bot._tasks.add(task)

    # Mock resource manager cleanup
    mock_cleanup = mocker.patch.object(terminal_bot._resource_manager, "force_cleanup")

    # Test cleanup
    await terminal_bot._cleanup()

    # Verify cleanup
    mock_cleanup.assert_called_once()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_stream_terminal_bot(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test streaming bot responses.

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
    """
    # Mock resource manager methods
    mock_check = mocker.patch.object(terminal_bot._resource_manager, "check_memory", return_value=True)
    mock_track = mocker.patch.object(terminal_bot._resource_manager, "track_task", autospec=True)
    mock_track.return_value = asyncio.Future()
    mock_track.return_value.set_result(None)
    mock_track_mem = mocker.patch.object(terminal_bot._resource_manager, "track_memory")
    mock_release = mocker.patch.object(terminal_bot._resource_manager, "release_memory")

    # Test streaming
    chunks = []
    async for chunk in terminal_bot.stream_terminal_bot("test prompt"):
        chunks.append(chunk)

    # Verify resource management
    mock_check.assert_called_once()
    mock_track.assert_called_once()
    assert mock_track_mem.call_count > 0
    assert mock_release.call_count > 0
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_stream_terminal_bot_memory_limit(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test streaming with memory limit exceeded.

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
    """
    # Mock memory check to fail
    mocker.patch.object(terminal_bot._resource_manager, "check_memory", return_value=False)

    # Test streaming with memory limit
    with pytest.raises(RuntimeError, match="Memory limit exceeded"):
        async for _ in terminal_bot.stream_terminal_bot("test prompt"):
            pass


@pytest.mark.asyncio
async def test_invoke_terminal_bot(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test invoking bot.

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
    """
    # Mock resource manager methods
    mock_check = mocker.patch.object(terminal_bot._resource_manager, "check_memory", return_value=True)
    mock_track = mocker.patch.object(terminal_bot._resource_manager, "track_task", autospec=True)
    mock_track.return_value = asyncio.Future()
    mock_track.return_value.set_result(None)

    # Test invocation
    response, steps = await terminal_bot.invoke_terminal_bot("test prompt")

    # Verify response and resource management
    assert response == "test prompt"
    assert steps == []
    assert mock_check.call_count == 2  # Called by both invoke and stream
    assert mock_track.call_count == 2  # Called by both invoke and stream


@pytest.mark.asyncio
async def test_invoke_terminal_bot_size_limit(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test invoking bot with response size limit.

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
    """
    # Set small response size limit
    terminal_bot._resource_manager.limits = ResourceLimits(
        max_memory_mb=128,
        max_tasks=5,
        max_response_size_mb=0.000001,  # Very small limit
        max_buffer_size_kb=32,
        task_timeout_seconds=1,
    )

    # Create large test prompt
    large_prompt = "x" * 1024 * 1024  # 1MB

    # Test invocation with size limit
    with pytest.raises(RuntimeError, match="Response size exceeds limit"):
        await terminal_bot.invoke_terminal_bot(large_prompt)


@pytest.mark.asyncio
async def test_start_and_shutdown(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test bot startup and shutdown.

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
    """
    # Mock cleanup
    mock_cleanup = mocker.patch.object(terminal_bot, "_cleanup")

    # Start bot in background
    task = asyncio.create_task(terminal_bot.start())

    # Trigger shutdown
    terminal_bot._shutdown_event.set()
    await task

    # Verify cleanup
    mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_context_manager(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test async context manager.

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
    """
    # Mock cleanup
    mock_cleanup = mocker.patch.object(terminal_bot, "_cleanup")

    # Test context manager
    async with terminal_bot as bot:
        assert bot is terminal_bot

    # Verify cleanup
    mock_cleanup.assert_called_once()
