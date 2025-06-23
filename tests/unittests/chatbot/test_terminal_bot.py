"""Tests for the terminal bot module."""

from __future__ import annotations

import asyncio
import datetime
import signal
import sys

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

import pytest_structlog
import structlog

from langchain.schema import AIMessage, HumanMessage

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

    This test verifies:
    - Resource management (memory tracking, task tracking)
    - Stream chunk processing
    - Proper cleanup
    - Logging of events
    - Timeout handling
    - Message formatting

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
    """
    # Mock aiosettings to enable resource management
    mocker.patch("democracy_exe.chatbot.terminal_bot.aiosettings.enable_resource_management", True)

    # Mock resource manager methods with async mocks
    mock_check = mocker.patch.object(
        terminal_bot._resource_manager, "check_memory", new_callable=mocker.AsyncMock, return_value=True
    )
    mock_track = mocker.patch.object(
        terminal_bot._resource_manager, "track_task", new_callable=mocker.AsyncMock, return_value=None
    )
    mock_track_mem = mocker.patch.object(
        terminal_bot._resource_manager, "track_memory", new_callable=mocker.AsyncMock, return_value=None
    )
    mock_release = mocker.patch.object(
        terminal_bot._resource_manager, "release_memory", new_callable=mocker.AsyncMock, return_value=None
    )

    # Mock message handler with async mocks
    test_chunks = ["Hello", " World", "!"]
    mock_message = mocker.patch.object(
        terminal_bot._message_handler, "format_message", new_callable=mocker.AsyncMock, return_value="test prompt"
    )
    mock_input_dict = mocker.patch.object(
        terminal_bot._message_handler,
        "create_input_dict",
        new_callable=mocker.AsyncMock,
        return_value={"messages": [{"role": "user", "content": "test prompt"}]},
    )

    # Mock graph
    mock_graph = mocker.Mock()
    mock_graph.stream = mocker.Mock(return_value=[{"messages": [{"content": chunk}]} for chunk in test_chunks])

    # Mock stream handler's process_stream method
    async def mock_process_stream(*args, **kwargs):
        """Mock process_stream that tracks memory and yields chunks."""
        # Track memory at start of streaming
        await mock_track_mem()
        structlog.get_logger().info("Processing stream data")

        async for event in mock_graph.stream(None, None, stream_mode="values"):
            if terminal_bot._stream_handler._interrupt_event.is_set():
                structlog.get_logger().info("Stream interrupted")
                break
            yield str(event["messages"][-1].content)

    # Create a mock stream handler class
    class MockStreamHandler:
        def __init__(self):
            self._interrupt_event = asyncio.Event()

        def interrupt(self):
            self._interrupt_event.set()

        async def process_stream(self, *args, **kwargs):
            async for chunk in mock_process_stream(*args, **kwargs):
                yield chunk

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            await mock_release()

    # Replace the stream handler with our mock
    terminal_bot._stream_handler = MockStreamHandler()

    # Test streaming with structlog capture and timeout
    with structlog.testing.capture_logs() as captured:
        chunks = []
        try:
            async with asyncio.timeout(2.0):  # 2 second timeout
                async for chunk in terminal_bot.stream_terminal_bot("test prompt", graph=mock_graph):
                    chunks.append(chunk)
                    # Verify each chunk is a string
                    assert isinstance(chunk, str), f"Expected string chunk, got {type(chunk)}"
        except TimeoutError:
            pytest.fail("Streaming timed out")

        # Verify resource management
        mock_check.assert_called_once()
        mock_track.assert_called_once()
        mock_track_mem.assert_called_once()  # Memory tracked once at start
        mock_release.assert_called_once()  # Memory released once at end

        # Verify chunks were received
        assert chunks == test_chunks, f"Expected chunks {test_chunks}, got {chunks}"

        # Verify logging
        assert any(log.get("event") == "Processing stream data" for log in captured), (
            "Expected 'Processing stream data' event not found in logs"
        )

        # Verify task cleanup
        current_task = asyncio.current_task()
        assert current_task not in terminal_bot._tasks, "Task not properly cleaned up"


# @pytest.mark.asyncio
# async def test_stream_terminal_bot_memory_limit(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
#     """Test streaming with memory limit exceeded.

#     Args:
#         terminal_bot: Test terminal bot
#         mocker: Pytest mocker
#     """
#     # Mock memory check to fail
#     mocker.patch.object(terminal_bot._resource_manager, "check_memory", return_value=False)

#     # Test streaming with memory limit
#     with pytest.raises(RuntimeError, match="Memory limit exceeded"):
#         async for _ in terminal_bot.stream_terminal_bot("test prompt"):
#             pass


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
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


# @pytest.mark.asyncio
# async def test_invoke_terminal_bot_size_limit(terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture) -> None:
#     """Test invoking bot with response size limit.

#     Args:
#         terminal_bot: Test terminal bot
#         mocker: Pytest mocker
#     """
#     # Set small response size limit
#     terminal_bot._resource_manager.limits = ResourceLimits(
#         max_memory_mb=128,
#         max_tasks=5,
#         max_response_size_mb=0.000001,  # Very small limit
#         max_buffer_size_kb=32,
#         task_timeout_seconds=1,
#     )

#     # Create large test prompt
#     large_prompt = "x" * 1024 * 1024  # 1MB

#     # Test invocation with size limit
#     with pytest.raises(RuntimeError, match="Response size exceeds limit"):
#         await terminal_bot.invoke_terminal_bot(large_prompt)


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


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
@pytest.mark.asyncio
async def test_stream_terminal_bot_interruptible(
    terminal_bot: ThreadSafeTerminalBot, mocker: MockerFixture, log: pytest_structlog.StructuredLogCapture
) -> None:
    """Test interruptible streaming bot responses.

    This test verifies:
    - Stream can be interrupted
    - Resources are cleaned up after interruption
    - Proper event logging
    - Task cleanup

    Args:
        terminal_bot: Test terminal bot
        mocker: Pytest mocker
        log: Structured log capture fixture
    """
    # Mock aiosettings to enable resource management
    mocker.patch("democracy_exe.chatbot.terminal_bot.aiosettings.enable_resource_management", True)

    # Mock resource manager methods with async mocks
    mock_check = mocker.patch.object(
        terminal_bot._resource_manager, "check_memory", new_callable=mocker.AsyncMock, return_value=True
    )
    mock_track = mocker.patch.object(
        terminal_bot._resource_manager, "track_task", new_callable=mocker.AsyncMock, return_value=None
    )
    mock_track_mem = mocker.patch.object(
        terminal_bot._resource_manager, "track_memory", new_callable=mocker.AsyncMock, return_value=None
    )
    mock_release = mocker.patch.object(
        terminal_bot._resource_manager, "release_memory", new_callable=mocker.AsyncMock, return_value=None
    )

    # Mock message handler with async mocks
    test_chunks = ["Hello", " World", "!"]
    mock_message = mocker.patch.object(
        terminal_bot._message_handler,
        "format_message",
        new_callable=mocker.AsyncMock,
        return_value=HumanMessage(content="test prompt"),
    )
    mock_input_dict = mocker.patch.object(
        terminal_bot._message_handler,
        "create_input_dict",
        new_callable=mocker.AsyncMock,
        return_value={"messages": [HumanMessage(content="test prompt")]},
    )

    # Mock graph with proper state handling
    thread = {"configurable": {"thread_id": "1"}}
    mock_graph = mocker.Mock()

    async def mock_stream(*args, **kwargs):
        """Mock stream that yields events with proper state."""
        for chunk in test_chunks:
            yield {"messages": [AIMessage(content=chunk)], "next": ("continue",)}

    mock_graph.stream = mock_stream

    # Create a mock stream handler class
    class MockStreamHandler:
        """Mock stream handler for testing."""

        def __init__(self):
            """Initialize mock stream handler."""
            self._interrupt_event = asyncio.Event()

        def interrupt(self):
            """Set the interrupt event."""
            self._interrupt_event.set()

        async def process_stream(self, *args, **kwargs):
            """Process the stream and yield chunks.

            Args:
                *args: Positional arguments
                **kwargs: Keyword arguments

            Yields:
                str: Processed chunks
            """
            try:
                # Track memory at start of streaming
                await mock_track_mem()
                log.info("Processing stream data")

                async for event in mock_stream():
                    if self._interrupt_event.is_set():
                        log.info("Stream interrupted")
                        break
                    yield str(event["messages"][-1].content)
            finally:
                await mock_release()

        async def __aenter__(self):
            """Enter async context.

            Returns:
                MockStreamHandler: This instance
            """
            return self

        async def __aexit__(self, *args):
            """Exit async context.

            Args:
                *args: Context manager arguments
            """
            pass

    # Replace the stream handler with our mock
    terminal_bot._stream_handler = MockStreamHandler()

    # Test streaming with timeout
    chunks = []
    try:
        async with asyncio.timeout(2.0):  # 2 second timeout
            async for chunk in terminal_bot.stream_terminal_bot("test prompt", graph=mock_graph, interruptable=True):
                chunks.append(chunk)
                if len(chunks) >= 2:  # Interrupt after 2 chunks
                    terminal_bot._stream_handler.interrupt()
                    break
    except TimeoutError:
        pytest.fail("Streaming timed out")

    # Verify resource management
    mock_check.assert_called_once()
    mock_track.assert_called_once()
    mock_track_mem.assert_called_once()  # Memory tracked once at start
    mock_release.assert_called_once()  # Memory released once at end

    # Verify chunks received before interruption
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    assert chunks == test_chunks[:2], f"Expected chunks {test_chunks[:2]}, got {chunks}"

    # Verify logging
    assert log.has("Processing stream data", level="info"), "Expected 'Processing stream data' event not found in logs"
    assert log.has("Stream interrupted", level="info"), "Expected 'Stream interrupted' event not found in logs"

    # Verify task cleanup
    current_task = asyncio.current_task()
    assert current_task not in terminal_bot._tasks, "Task not properly cleaned up"
