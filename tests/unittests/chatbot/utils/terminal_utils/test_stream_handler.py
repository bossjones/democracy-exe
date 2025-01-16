"""Unit tests for the terminal stream handler."""

from __future__ import annotations

import asyncio

from collections.abc import AsyncGenerator
from typing import Any, Dict, List, cast

import structlog

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

import pytest

from pytest_mock import MockerFixture

from democracy_exe.chatbot.utils.terminal_utils import StreamHandler
from democracy_exe.chatbot.utils.terminal_utils.message_handler import MessageHandler


class MockGraph:
    """Mock graph for testing."""

    def __init__(self, responses: list[dict[str, list[BaseMessage]]]) -> None:
        """Initialize mock graph.

        Args:
            responses: List of response dictionaries
        """
        self.responses = responses

    def stream(
        self, user_input: Any = None, config: Any = None, stream_mode: str = "values"
    ) -> list[dict[str, list[BaseMessage]]]:
        """Mock stream method.

        Args:
            user_input: User input
            config: Configuration
            stream_mode: Stream mode

        Returns:
            List of response dictionaries
        """
        return self.responses


@pytest.fixture
def stream_handler() -> StreamHandler:
    """Create a stream handler instance for testing.

    Returns:
        StreamHandler: Test stream handler
    """
    return StreamHandler()


@pytest.fixture
def mock_graph() -> MockGraph:
    """Create a mock graph for testing.

    Returns:
        MockGraph: Test mock graph
    """
    responses = [{"messages": [AIMessage(content="Response 1")]}, {"messages": [AIMessage(content="Response 2")]}]
    return MockGraph(responses)


@pytest.mark.asyncio
async def test_process_stream(stream_handler: StreamHandler, mock_graph: MockGraph) -> None:
    """Test stream processing.

    Args:
        stream_handler: Test stream handler
        mock_graph: Test mock graph
    """
    with structlog.testing.capture_logs() as captured:
        user_input = {"messages": [HumanMessage(content="Test input")]}
        chunks = []
        async for chunk in stream_handler.process_stream(mock_graph, user_input):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == "Response 1"
        assert chunks[1] == "Response 2"

        # Verify logging
        assert any(log.get("event") == "Processing stream" for log in captured)


@pytest.mark.asyncio
async def test_process_stream_basic_config(stream_handler: StreamHandler, mocker: MockerFixture) -> None:
    """Test stream processing with basic RunnableConfig.

    Args:
        stream_handler: Test stream handler
        mocker: Pytest mocker fixture
    """
    # Create mock message handler
    mock_message_handler = mocker.Mock(spec=MessageHandler)

    async def mock_stream_chunks(chunks):
        for chunk in chunks:
            if chunk.get("messages"):
                yield chunk["messages"][0].content

    mock_message_handler.stream_chunks = mock_stream_chunks
    stream_handler._message_handler = mock_message_handler

    # Create mock graph
    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    mock_graph.stream.return_value = [{"messages": [AIMessage(content="Response 1")]}]

    # Test with basic config
    config = RunnableConfig(tags=["test"], metadata={"test": True})
    user_input = {"messages": [HumanMessage(content="Test input")]}

    # Mock the logger
    mock_logger = mocker.Mock()
    stream_handler._logger = mock_logger

    # Trigger an error to test error logging
    mock_graph.stream.side_effect = ValueError("Test error")

    # Process stream should raise the error
    with pytest.raises(ValueError, match="Test error"):
        async for _ in stream_handler.process_stream(mock_graph, user_input, config=config):
            pass

    # Verify error was logged
    mock_logger.error.assert_called_once_with("Error processing stream", error="Test error")


@pytest.mark.asyncio
async def test_process_stream_with_callbacks(stream_handler: StreamHandler, mocker: MockerFixture) -> None:
    """Test stream processing with callbacks in config.

    Args:
        stream_handler: Test stream handler
        mocker: Pytest mocker fixture
    """
    # Mock callback
    callback_called = False

    def test_callback(event):
        nonlocal callback_called
        callback_called = True

    # Setup mocks
    mock_message_handler = mocker.Mock(spec=MessageHandler)

    async def mock_stream_chunks(chunks):
        for chunk in chunks:
            if chunk.get("messages"):
                yield chunk["messages"][0].content

    mock_message_handler.stream_chunks = mock_stream_chunks
    stream_handler._message_handler = mock_message_handler

    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    mock_graph.stream.return_value = [{"messages": [AIMessage(content="Response 1")]}]

    # Test with callback config
    config = RunnableConfig(callbacks=[test_callback])
    user_input = {"messages": [HumanMessage(content="Test input")]}

    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input, config=config):
        chunks.append(chunk)

    # Verify callback was used
    assert callback_called, "Callback was not triggered"
    assert len(chunks) == 1
    assert chunks[0] == "Response 1"


@pytest.mark.asyncio
async def test_process_stream_with_recursion_limit(stream_handler: StreamHandler, mocker: MockerFixture) -> None:
    """Test stream processing with recursion limit in config.

    Args:
        stream_handler: Test stream handler
        mocker: Pytest mocker fixture
    """
    # Setup mocks
    mock_message_handler = mocker.Mock(spec=MessageHandler)

    async def mock_stream_chunks(chunks):
        for chunk in chunks:
            if chunk.get("messages"):
                yield chunk["messages"][0].content

    mock_message_handler.stream_chunks = mock_stream_chunks
    stream_handler._message_handler = mock_message_handler

    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    mock_graph.stream.return_value = [{"messages": [AIMessage(content="Response 1")]}]

    # Test with recursion limit
    config = RunnableConfig(recursion_limit=10)
    user_input = {"messages": [HumanMessage(content="Test input")]}

    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input, config=config):
        chunks.append(chunk)

    # Verify config was used correctly
    mock_graph.stream.assert_called_once_with(user_input, config, stream_mode="values")
    assert len(chunks) == 1
    assert chunks[0] == "Response 1"


@pytest.mark.asyncio
async def test_process_stream_config_validation(stream_handler: StreamHandler, mocker: MockerFixture) -> None:
    """Test stream processing with invalid config values.

    Args:
        stream_handler: Test stream handler
        mocker: Pytest mocker fixture
    """
    # Setup mocks
    mock_message_handler = mocker.Mock(spec=MessageHandler)

    async def mock_stream_chunks(chunks):
        for chunk in chunks:
            if chunk.get("messages"):
                yield chunk["messages"][0].content

    mock_message_handler.stream_chunks = mock_stream_chunks
    stream_handler._message_handler = mock_message_handler

    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    mock_graph.stream.return_value = [{"messages": [AIMessage(content="Response 1")]}]

    # Test with invalid config values
    invalid_configs = [
        RunnableConfig(recursion_limit=-1),  # Invalid recursion limit
        RunnableConfig(tags=123),  # Invalid tags type
        RunnableConfig(metadata=["invalid"]),  # Invalid metadata type
    ]

    for config in invalid_configs:
        with pytest.raises((ValueError, TypeError)):
            user_input = {"messages": [HumanMessage(content="Test input")]}
            async for _ in stream_handler.process_stream(mock_graph, user_input, config=config):
                pass


@pytest.mark.asyncio
async def test_process_empty_stream(stream_handler: StreamHandler) -> None:
    """Test processing empty stream responses.

    Args:
        stream_handler: Test stream handler
    """
    empty_graph = MockGraph([])
    user_input = {"messages": [HumanMessage(content="Test input")]}

    chunks = []
    async for chunk in stream_handler.process_stream(empty_graph, user_input):
        chunks.append(chunk)

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_process_malformed_messages(stream_handler: StreamHandler) -> None:
    """Test handling of malformed message responses.

    Args:
        stream_handler: Test stream handler
    """
    # Create graph with malformed responses
    malformed_responses = [
        {"wrong_key": []},  # Missing messages key
        {"messages": []},  # Empty messages list
        {"messages": [{"content": "Invalid message"}]},  # Not a BaseMessage
    ]
    malformed_graph = MockGraph(malformed_responses)
    user_input = {"messages": [HumanMessage(content="Test input")]}

    chunks = []
    async for chunk in stream_handler.process_stream(malformed_graph, user_input):
        chunks.append(chunk)

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_process_stream_interruptible(
    stream_handler: StreamHandler, mock_graph: MockGraph, mocker: MockerFixture
) -> None:
    """Test interruptible stream processing.

    Args:
        stream_handler: Test stream handler
        mock_graph: Test mock graph
        mocker: Pytest mocker fixture
    """
    with structlog.testing.capture_logs() as captured:
        # Mock user input to approve continuation
        mocker.patch("asyncio.to_thread", return_value="y")

        user_input = {"messages": [HumanMessage(content="Test input")]}
        stream_handler.interrupt()  # Trigger interruption

        chunks = []
        async for chunk in stream_handler.process_stream(mock_graph, user_input, interruptable=True):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == "Response 1"
        assert chunks[1] == "Response 2"

        # Verify interruption logging
        assert any(log.get("event") == "Stream interrupted" for log in captured)


@pytest.mark.asyncio
async def test_process_stream_interrupt_cancel(
    stream_handler: StreamHandler, mock_graph: MockGraph, mocker: MockerFixture
) -> None:
    """Test stream interruption and cancellation.

    Args:
        stream_handler: Test stream handler
        mock_graph: Test mock graph
        mocker: Pytest mocker fixture
    """
    with structlog.testing.capture_logs() as captured:
        # Mock user input to cancel continuation
        mocker.patch("asyncio.to_thread", return_value="n")

        user_input = {"messages": [HumanMessage(content="Test input")]}
        stream_handler.interrupt()  # Trigger interruption

        chunks = []
        async for chunk in stream_handler.process_stream(mock_graph, user_input, interruptable=True):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0] == "Response 1"
        assert chunks[1] == "Operation cancelled by user."

        # Verify cancellation logging
        assert any(log.get("event") == "Stream cancelled by user" for log in captured)


@pytest.mark.asyncio
async def test_process_stream_error(
    stream_handler: StreamHandler, mock_graph: MockGraph, mocker: MockerFixture
) -> None:
    """Test stream processing error handling.

    Args:
        stream_handler: Test stream handler
        mock_graph: Test mock graph
        mocker: Pytest mocker fixture
    """
    with structlog.testing.capture_logs() as captured:

        def mock_stream(*args: Any, **kwargs: Any) -> None:
            raise ValueError("Test error")

        mocker.patch.object(mock_graph, "stream", side_effect=mock_stream)
        user_input = {"messages": [HumanMessage(content="Test input")]}

        with pytest.raises(ValueError, match="Test error"):
            async for _ in stream_handler.process_stream(mock_graph, user_input):
                pass

        # Verify error logging
        assert any(log.get("event") == "Error processing stream" for log in captured)


@pytest.mark.asyncio
async def test_stream_handler_context(stream_handler: StreamHandler) -> None:
    """Test stream handler context management.

    Args:
        stream_handler: Test stream handler
    """
    with structlog.testing.capture_logs() as captured:
        async with stream_handler as handler:
            assert not handler._interrupt_event.is_set()
            handler.interrupt()
            assert handler._interrupt_event.is_set()

        # Verify reset after context exit
        assert not stream_handler._interrupt_event.is_set()

        # Verify context logging
        assert any(log.get("event") == "Stream handler initialized" for log in captured)
        assert any(log.get("event") == "Stream handler reset" for log in captured)


@pytest.mark.asyncio
async def test_message_handler_interaction(stream_handler: StreamHandler, mocker: MockerFixture) -> None:
    """Test interaction with message handler.

    Args:
        stream_handler: Test stream handler
        mocker: Pytest mocker fixture
    """
    # Mock message handler
    mock_message_handler = mocker.Mock(spec=MessageHandler)
    mock_message_handler.stream_chunks.return_value = AsyncGenerator[str, None](
        lambda: (yield from ["Chunk 1", "Chunk 2"])
    )

    # Replace message handler
    stream_handler._message_handler = mock_message_handler

    # Test stream processing
    mock_graph = MockGraph([{"messages": [AIMessage(content="Test")]}])
    user_input = {"messages": [HumanMessage(content="Test input")]}

    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks == ["Chunk 1", "Chunk 2"]
    mock_message_handler.stream_chunks.assert_called_once()


@pytest.mark.asyncio
async def test_cleanup_on_error(stream_handler: StreamHandler, mock_graph: MockGraph) -> None:
    """Test cleanup when error occurs during streaming.

    Args:
        stream_handler: Test stream handler
        mock_graph: Test mock graph
    """
    with structlog.testing.capture_logs() as captured:
        try:
            async with stream_handler:
                stream_handler.interrupt()
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify proper cleanup
        assert not stream_handler._interrupt_event.is_set()
        assert any(log.get("event") == "Stream handler reset" for log in captured)
