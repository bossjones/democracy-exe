# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
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

    This test verifies:
    1. Callbacks are called with correct parameters
    2. Callbacks are called at the right time
    3. Multiple callbacks work correctly
    4. Callback errors are handled properly

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

    # Create mock callbacks
    callback1 = mocker.Mock(name="callback1")
    callback2 = mocker.Mock(name="callback2")
    error_callback = mocker.Mock(name="error_callback", side_effect=ValueError("Callback error"))

    # Create mock graph with callback handling
    mock_graph = mocker.Mock(spec=CompiledStateGraph)

    def mock_stream(user_input, config=None, stream_mode="values"):
        responses = [{"messages": [AIMessage(content="Response 1")]}, {"messages": [AIMessage(content="Response 2")]}]
        if config and isinstance(config, dict) and config.get("callbacks"):
            for response in responses:
                for callback in config["callbacks"]:
                    try:
                        callback(response)
                    except Exception:
                        pass  # Callbacks shouldn't affect stream
        return responses

    mock_graph.stream = mock_stream

    # Test with multiple callbacks
    config = RunnableConfig(callbacks=[callback1, callback2], tags=["test"], metadata={"test": True})
    user_input = {"messages": [HumanMessage(content="Test input")]}

    # Process stream
    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input, config=config):
        chunks.append(chunk)

    # Verify chunks
    assert len(chunks) == 2
    assert chunks == ["Response 1", "Response 2"]

    # Verify callbacks were called for each response
    assert callback1.call_count == 2, "First callback should be called twice"
    assert callback2.call_count == 2, "Second callback should be called twice"

    # Verify callback parameters
    for callback in [callback1, callback2]:
        calls = callback.call_args_list
        assert len(calls) == 2, "Each callback should be called twice"

        # First call should be for Response 1
        args1, _ = calls[0]
        assert len(args1) == 1, "Callback should receive one positional argument"
        assert args1[0]["messages"][0].content == "Response 1"

        # Second call should be for Response 2
        args2, _ = calls[1]
        assert len(args2) == 1, "Callback should receive one positional argument"
        assert args2[0]["messages"][0].content == "Response 2"

    # Test error callback handling
    config_with_error = RunnableConfig(callbacks=[error_callback])
    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input, config=config_with_error):
        chunks.append(chunk)

    # Verify chunks still processed despite callback error
    assert len(chunks) == 2
    assert chunks == ["Response 1", "Response 2"]

    # Verify error callback was called
    assert error_callback.call_count == 2, "Error callback should be called twice"

    # Test callback with empty response
    mock_graph.stream = lambda *args, **kwargs: []  # Override to return empty response
    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input, config=config):
        chunks.append(chunk)

    # Verify no additional callbacks for empty response
    assert len(chunks) == 0
    assert callback1.call_count == 2, "Callback count should not increase for empty response"
    assert callback2.call_count == 2, "Callback count should not increase for empty response"


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

    This test verifies:
    1. Invalid config values are properly handled
    2. Config is passed correctly to graph
    3. Errors are properly logged
    4. Different config types are supported

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

    # Create mock graph
    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    mock_graph.stream.return_value = [{"messages": [AIMessage(content="Response 1")]}]

    # Test different config types
    test_configs = [
        # Basic config
        RunnableConfig(tags=["test"], metadata={"test": True}),
        # Config with callbacks
        RunnableConfig(callbacks=[lambda x: None], tags=["test"]),
        # Config with recursion limit
        RunnableConfig(recursion_limit=10),
        # Config with all options
        RunnableConfig(callbacks=[lambda x: None], tags=["test"], metadata={"test": True}, recursion_limit=10),
        # Empty config
        RunnableConfig(),
        # Config with empty lists
        RunnableConfig(tags=[], callbacks=[], metadata={}),
    ]

    for config in test_configs:
        # Process stream with config
        user_input = {"messages": [HumanMessage(content="Test input")]}
        chunks = []
        async for chunk in stream_handler.process_stream(mock_graph, user_input, config=config):
            chunks.append(chunk)

        # Verify stream processed correctly
        assert len(chunks) == 1
        assert chunks[0] == "Response 1"

        # Verify config was passed to graph
        mock_graph.stream.assert_called_with(user_input, config, stream_mode="values")
        mock_graph.reset_mock()

    # Test error handling
    mock_graph.stream.side_effect = ValueError("Test error")

    with structlog.testing.capture_logs() as captured:
        with pytest.raises(ValueError, match="Test error"):
            async for _ in stream_handler.process_stream(mock_graph, user_input, config=test_configs[0]):
                pass

        # Print captured logs for debugging
        print("\nCaptured logs:")
        for log in captured:
            print(f"Log entry: {log}")

        # Verify error was logged
        assert any(
            log.get("event") == "Error processing stream" and "Test error" in str(log.get("error", ""))
            for log in captured
        ), "Error was not properly logged"


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

    This test verifies:
    1. Message handler processes chunks correctly
    2. Errors from message handler are handled
    3. Empty chunks are handled
    4. Logging is correct

    Args:
        stream_handler: Test stream handler
        mocker: Pytest mocker fixture
    """
    # Setup mock message handler
    mock_message_handler = mocker.Mock(spec=MessageHandler)

    # Create async generator class for chunks
    class AsyncChunkGenerator:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.chunks:
                raise StopAsyncIteration
            chunk = self.chunks.pop(0)
            if chunk.get("messages"):
                await asyncio.sleep(0)  # Simulate async behavior
                return chunk["messages"][0].content
            elif isinstance(chunk, str):
                await asyncio.sleep(0)  # Simulate async behavior
                return chunk
            return None

    def mock_stream_chunks(chunks):
        return AsyncChunkGenerator(list(chunks))

    mock_message_handler.stream_chunks = mock_stream_chunks
    stream_handler._message_handler = mock_message_handler

    # Create mock graph with different response types
    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    responses = [
        {"messages": [AIMessage(content="Response 1")]},
        {"messages": [AIMessage(content="Response 2")]},
        {"messages": []},  # Empty messages
        {"other_key": "value"},  # Wrong key
        {"messages": [AIMessage(content="Response 3")]},
    ]
    mock_graph.stream.return_value = responses

    # Test normal processing
    user_input = {"messages": [HumanMessage(content="Test input")]}
    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input):
        if chunk is not None:
            chunks.append(chunk)

    # Verify chunks
    assert len(chunks) == 3, "Should process valid messages only"
    assert chunks == ["Response 1", "Response 2", "Response 3"]

    # Test error in message handler
    class ErrorGenerator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(0)  # Simulate async behavior
            raise ValueError("Message handler error")

    mock_message_handler.stream_chunks = lambda chunks: ErrorGenerator()

    with structlog.testing.capture_logs() as captured:
        with pytest.raises(ValueError, match="Message handler error"):
            async for _ in stream_handler.process_stream(mock_graph, user_input):
                pass

        # Verify error was logged
        assert any(
            log.get("event") == "Error processing stream" and "Message handler error" in str(log.get("error", ""))
            for log in captured
        ), "Error was not properly logged"

    # Test empty response
    mock_graph.stream.return_value = []
    mock_message_handler.stream_chunks = mock_stream_chunks  # Reset to working version

    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input):
        if chunk is not None:
            chunks.append(chunk)

    assert len(chunks) == 0, "Should handle empty response"

    # Test direct string chunks
    mock_graph.stream.return_value = ["Direct 1", "Direct 2"]
    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input):
        if chunk is not None:
            chunks.append(chunk)

    assert len(chunks) == 2, "Should handle direct string chunks"
    assert chunks == ["Direct 1", "Direct 2"]


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
