"""Unit tests for the terminal stream handler."""

from __future__ import annotations

import asyncio

from collections.abc import AsyncGenerator
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

import pytest

from pytest_mock import MockerFixture

from democracy_exe.chatbot.utils.terminal_utils import StreamHandler


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
    user_input = {"messages": [HumanMessage(content="Test input")]}
    chunks = []
    async for chunk in stream_handler.process_stream(mock_graph, user_input):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0] == "Response 1"
    assert chunks[1] == "Response 2"


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

    def mock_stream(*args: Any, **kwargs: Any) -> None:
        raise ValueError("Test error")

    mocker.patch.object(mock_graph, "stream", side_effect=mock_stream)
    user_input = {"messages": [HumanMessage(content="Test input")]}

    with pytest.raises(ValueError, match="Test error"):
        async for _ in stream_handler.process_stream(mock_graph, user_input):
            pass


@pytest.mark.asyncio
async def test_stream_handler_context(stream_handler: StreamHandler) -> None:
    """Test stream handler context management.

    Args:
        stream_handler: Test stream handler
    """
    async with stream_handler as handler:
        assert not handler._interrupt_event.is_set()
        handler.interrupt()
        assert handler._interrupt_event.is_set()

    # Verify reset after context exit
    assert not stream_handler._interrupt_event.is_set()
