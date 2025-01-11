"""Tests for terminal_bot.py functionality."""

from __future__ import annotations

import asyncio
import gc
import signal
import sys
import threading
import weakref

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List, cast

import structlog

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

import pytest

from democracy_exe.chatbot.terminal_bot import BotState, ThreadSafeTerminalBot, invoke_terminal_bot, stream_terminal_bot


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
async def terminal_bot() -> AsyncGenerator[ThreadSafeTerminalBot, None]:
    """Create a ThreadSafeTerminalBot instance for testing.

    Yields:
        ThreadSafeTerminalBot: The bot instance for testing
    """
    bot = ThreadSafeTerminalBot()
    async with bot:
        yield bot


@pytest.fixture
def mock_graph(mocker: MockerFixture) -> CompiledStateGraph:
    """Create a mock CompiledStateGraph for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        CompiledStateGraph: The mocked graph instance
    """
    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    mock_graph.stream.return_value = [{"messages": [AIMessage(content="Test response")]}]
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="Test response")], "response": "Test response"}
    return mock_graph


@pytest.mark.asyncio
async def test_terminal_bot_initialization(terminal_bot: ThreadSafeTerminalBot) -> None:
    """Test ThreadSafeTerminalBot initialization.

    Args:
        terminal_bot: The bot instance to test
    """
    assert terminal_bot._state == BotState.RUNNING
    assert not terminal_bot._closed
    assert terminal_bot._loop is not None


@pytest.mark.asyncio
async def test_terminal_bot_cleanup(terminal_bot: ThreadSafeTerminalBot) -> None:
    """Test ThreadSafeTerminalBot cleanup.

    Args:
        terminal_bot: The bot instance to test
    """
    await terminal_bot.cleanup()
    assert terminal_bot._closed
    assert terminal_bot._state == BotState.CLOSED
    assert terminal_bot._loop is None


@pytest.mark.asyncio
async def test_terminal_bot_signal_handling(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test signal handling in ThreadSafeTerminalBot.

    Args:
        terminal_bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    mock_cleanup = mocker.patch.object(terminal_bot, "cleanup")
    terminal_bot._signal_handler(signal.SIGTERM, None)
    await asyncio.sleep(0)  # Allow event loop to process signal
    mock_cleanup.assert_called_once()


@pytest.mark.asyncio
async def test_stream_terminal_bot(mock_graph: CompiledStateGraph, caplog: LogCaptureFixture) -> None:
    """Test stream_terminal_bot function.

    Args:
        mock_graph: The mock graph instance
        caplog: Pytest log capture fixture
    """
    with structlog.testing.capture_logs() as captured:
        user_input = {"messages": [HumanMessage(content="Test input")]}
        stream_terminal_bot(mock_graph, user_input)

        # Verify logging
        assert any(log.get("event") == "Processing stream data" for log in captured), (
            "Expected 'Processing stream data' not found in logs"
        )

        # Verify graph interaction
        mock_graph.stream.assert_called_once()


@pytest.mark.asyncio
async def test_stream_terminal_bot_size_limit(mock_graph: CompiledStateGraph, mocker: MockerFixture) -> None:
    """Test size limits in stream_terminal_bot.

    Args:
        mock_graph: The mock graph instance
        mocker: Pytest mocker fixture
    """
    # Create large response
    large_content = "x" * (1024 * 1024 + 1)  # Exceeds 1MB limit
    mock_graph.stream.return_value = [{"messages": [AIMessage(content=large_content)]}]

    with pytest.raises(RuntimeError, match="Response size .* exceeds limit"):
        stream_terminal_bot(mock_graph, {"messages": [HumanMessage(content="Test")]})


@pytest.mark.asyncio
async def test_invoke_terminal_bot(mock_graph: CompiledStateGraph, caplog: LogCaptureFixture) -> None:
    """Test invoke_terminal_bot function.

    Args:
        mock_graph: The mock graph instance
        caplog: Pytest log capture fixture
    """
    with structlog.testing.capture_logs() as captured:
        response = invoke_terminal_bot(mock_graph, {"messages": [HumanMessage(content="Test")]})
        assert response == "Test response"

        # Verify logging
        assert any(log.get("event") == "AI response" for log in captured), "Expected 'AI response' not found in logs"


@pytest.mark.asyncio
async def test_invoke_terminal_bot_error(mock_graph: CompiledStateGraph, mocker: MockerFixture) -> None:
    """Test error handling in invoke_terminal_bot.

    Args:
        mock_graph: The mock graph instance
        mocker: Pytest mocker fixture
    """
    mock_graph.invoke.side_effect = Exception("Test error")

    with pytest.raises(Exception, match="Test error"):
        invoke_terminal_bot(mock_graph, {"messages": [HumanMessage(content="Test")]})


@pytest.mark.asyncio
async def test_terminal_bot_process_message(
    terminal_bot: ThreadSafeTerminalBot, mock_graph: CompiledStateGraph
) -> None:
    """Test message processing in ThreadSafeTerminalBot.

    Args:
        terminal_bot: The bot instance to test
        mock_graph: The mock graph instance
    """
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    await terminal_bot.process_message(mock_graph, "Test message", config)

    # Verify graph interaction
    mock_graph.stream.assert_called_once()


@pytest.mark.asyncio
async def test_terminal_bot_process_message_error(
    terminal_bot: ThreadSafeTerminalBot, mock_graph: CompiledStateGraph
) -> None:
    """Test error handling in ThreadSafeTerminalBot message processing.

    Args:
        terminal_bot: The bot instance to test
        mock_graph: The mock graph instance
    """
    mock_graph.stream.side_effect = Exception("Test error")
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    with pytest.raises(Exception, match="Test error"):
        await terminal_bot.process_message(mock_graph, "Test message", config)


@pytest.mark.asyncio
async def test_terminal_bot_closed_error(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
    """Test error handling when bot is closed.

    Args:
        terminal_bot: The bot instance to test
        mocker: Pytest mocker fixture
    """
    await terminal_bot.cleanup()

    with pytest.raises(RuntimeError, match="Bot is in closed state"):
        await terminal_bot.process_message(
            mocker.Mock(), "Test message", {"configurable": {"thread_id": "1", "user_id": "1"}}
        )


@pytest.mark.asyncio
async def test_terminal_bot_concurrent_tasks(
    terminal_bot: ThreadSafeTerminalBot, mock_graph: CompiledStateGraph
) -> None:
    """Test concurrent task handling in ThreadSafeTerminalBot.

    Args:
        terminal_bot: The bot instance to test
        mock_graph: The mock graph instance
    """
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    tasks = []

    # Create multiple concurrent tasks
    for _ in range(3):
        task = asyncio.create_task(terminal_bot.process_message(mock_graph, "Test message", config))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Verify all tasks completed
    assert all(task.done() for task in tasks)
