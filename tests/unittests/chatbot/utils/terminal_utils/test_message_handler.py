# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Unit tests for the terminal message handler."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

import pytest

from pytest_mock import MockerFixture

from democracy_exe.chatbot.utils.terminal_utils import MessageHandler


@pytest.fixture
def message_handler() -> MessageHandler:
    """Create a message handler instance for testing.

    Returns:
        MessageHandler: Test message handler
    """
    return MessageHandler()


@pytest.mark.asyncio
async def test_format_message(message_handler: MessageHandler) -> None:
    """Test message formatting.

    Args:
        message_handler: Test message handler
    """
    content = "Test message"
    message = await message_handler.format_message(content)
    assert isinstance(message, HumanMessage)
    assert message.content == content


@pytest.mark.asyncio
async def test_format_response_ai_message(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test AI message response formatting.

    Args:
        message_handler: Test message handler
        mocker: Pytest mocker fixture
    """
    mock_rprint = mocker.patch("democracy_exe.chatbot.utils.terminal_utils.message_handler.rprint")
    message = AIMessage(content="Test response")
    await message_handler.format_response(message)
    mock_rprint.assert_called_once_with("[bold blue]AI:[/bold blue] Test response")


@pytest.mark.asyncio
async def test_format_response_other_message(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test non-AI message response formatting.

    Args:
        message_handler: Test message handler
        mocker: Pytest mocker fixture
    """
    message = HumanMessage(content="Test message")
    mock_pretty_print = mocker.patch.object(message, "pretty_print")
    await message_handler.format_response(message)
    mock_pretty_print.assert_called_once()


@pytest.mark.asyncio
async def test_create_input_dict(message_handler: MessageHandler) -> None:
    """Test input dictionary creation.

    Args:
        message_handler: Test message handler
    """
    message = HumanMessage(content="Test message")
    input_dict = await message_handler.create_input_dict(message)
    assert "messages" in input_dict
    assert len(input_dict["messages"]) == 1
    assert input_dict["messages"][0] == message


@pytest.mark.asyncio
async def test_stream_chunks(message_handler: MessageHandler) -> None:
    """Test chunk streaming.

    Args:
        message_handler: Test message handler
    """

    async def mock_chunks():
        yield {"messages": [AIMessage(content="Chunk 1")]}
        yield {"messages": [AIMessage(content="Chunk 2")]}

    chunks = []
    async for chunk in message_handler.stream_chunks(mock_chunks()):
        chunks.append(chunk)

    assert len(chunks) == 2
    assert chunks[0] == "Chunk 1"
    assert chunks[1] == "Chunk 2"


@pytest.mark.asyncio
async def test_stream_chunks_error(message_handler: MessageHandler) -> None:
    """Test chunk streaming error handling.

    Args:
        message_handler: Test message handler
    """

    async def mock_chunks():
        yield {"messages": [AIMessage(content="Chunk 1")]}
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        async for _ in message_handler.stream_chunks(mock_chunks()):
            pass
