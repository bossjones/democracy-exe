"""Tests for message_handler.py functionality."""

from __future__ import annotations

import asyncio
import gc
import re

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List, cast

import discord
import structlog

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

import pytest

from democracy_exe.chatbot.handlers.message_handler import MessageHandler


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_bot(mocker: MockerFixture) -> Any:
    """Create a mock bot instance.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Any: The mock bot instance
    """
    return mocker.Mock()


@pytest.fixture
def message_handler(mock_bot: Any) -> MessageHandler:
    """Create a MessageHandler instance for testing.

    Args:
        mock_bot: The mock bot instance

    Returns:
        MessageHandler: The handler instance for testing
    """
    return MessageHandler(mock_bot)


@pytest.fixture
def mock_message(mocker: MockerFixture) -> discord.Message:
    """Create a mock Discord message.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        discord.Message: The mock message instance
    """
    message = mocker.Mock(spec=discord.Message)
    message.content = "Test message"
    message.attachments = []
    return message


@pytest.fixture
def mock_attachment(mocker: MockerFixture) -> discord.Attachment:
    """Create a mock Discord attachment.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        discord.Attachment: The mock attachment instance
    """
    attachment = mocker.Mock(spec=discord.Attachment)
    attachment.size = 1024  # 1KB
    attachment.filename = "test.txt"
    attachment.content_type = "text/plain"
    return attachment


@pytest.mark.asyncio
async def test_check_for_attachments_no_attachments(
    message_handler: MessageHandler, mock_message: discord.Message
) -> None:
    """Test check_for_attachments with no attachments.

    Args:
        message_handler: The handler instance to test
        mock_message: The mock message instance
    """
    result = await message_handler.check_for_attachments(mock_message)
    assert result == "Test message"


@pytest.mark.asyncio
async def test_check_for_attachments_size_limit(
    message_handler: MessageHandler, mock_message: discord.Message, mock_attachment: discord.Attachment
) -> None:
    """Test attachment size limits.

    Args:
        message_handler: The handler instance to test
        mock_message: The mock message instance
        mock_attachment: The mock attachment instance
    """
    mock_attachment.size = 100 * 1024 * 1024  # 100MB
    mock_message.attachments = [mock_attachment]

    with pytest.raises(RuntimeError, match="Total attachment size .* exceeds .* limit"):
        await message_handler.check_for_attachments(mock_message)


@pytest.mark.asyncio
async def test_check_for_attachments_image_url(
    message_handler: MessageHandler, mock_message: discord.Message, mocker: MockerFixture
) -> None:
    """Test image URL handling.

    Args:
        message_handler: The handler instance to test
        mock_message: The mock message instance
        mocker: Pytest mocker fixture
    """
    mock_message.content = "https://example.com/test.jpg"
    mocker.patch.object(message_handler.attachment_handler, "download_image", return_value=mocker.Mock())

    result = await message_handler.check_for_attachments(mock_message)
    assert "https://example.com/test.jpg" in result


@pytest.mark.asyncio
async def test_check_for_attachments_too_many_urls(
    message_handler: MessageHandler, mock_message: discord.Message
) -> None:
    """Test handling of too many image URLs.

    Args:
        message_handler: The handler instance to test
        mock_message: The mock message instance
    """
    urls = " ".join([f"https://example.com/test{i}.jpg" for i in range(6)])
    mock_message.content = urls

    with pytest.raises(ValueError, match="Too many image URLs in message"):
        await message_handler.check_for_attachments(mock_message)


@pytest.mark.asyncio
async def test_check_for_attachments_concurrent_downloads(
    message_handler: MessageHandler, mock_message: discord.Message, mocker: MockerFixture
) -> None:
    """Test concurrent download handling.

    Args:
        message_handler: The handler instance to test
        mock_message: The mock message instance
        mocker: Pytest mocker fixture
    """
    # Create multiple image URLs
    urls = " ".join([f"https://example.com/test{i}.jpg" for i in range(3)])
    mock_message.content = urls

    # Mock download_image to simulate delay
    async def slow_download(*args: Any) -> Any:
        await asyncio.sleep(0.1)
        return mocker.Mock()

    mocker.patch.object(message_handler.attachment_handler, "download_image", side_effect=slow_download)

    # Start multiple downloads concurrently
    tasks = []
    for _ in range(3):
        task = asyncio.create_task(message_handler.check_for_attachments(mock_message))
        tasks.append(task)

    # Verify downloads are properly limited
    start_time = asyncio.get_event_loop().time()
    await asyncio.gather(*tasks)
    elapsed = asyncio.get_event_loop().time() - start_time

    # Should take at least 0.1 seconds due to semaphore limiting concurrent downloads
    assert elapsed >= 0.1


@pytest.mark.asyncio
async def test_stream_bot_response(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test stream_bot_response function.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="Test response")]}

    input_data = {"messages": [HumanMessage(content="Test input")]}
    response = await message_handler.stream_bot_response(mock_graph, input_data)

    assert response == "Test response"
    mock_graph.invoke.assert_called_once_with(input_data)


@pytest.mark.asyncio
async def test_stream_bot_response_timeout(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test timeout handling in stream_bot_response.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_graph = mocker.Mock(spec=CompiledStateGraph)

    async def slow_invoke(*args: Any) -> dict[str, list[AIMessage]]:
        await asyncio.sleep(31)  # Longer than timeout
        return {"messages": [AIMessage(content="Should not reach here")]}

    mock_graph.ainvoke = mocker.AsyncMock(side_effect=slow_invoke)

    with pytest.raises(RuntimeError, match="Response generation timed out"):
        await message_handler.stream_bot_response(mock_graph, {"messages": [HumanMessage(content="Test input")]})


@pytest.mark.asyncio
async def test_stream_bot_response_size_limit(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test response size limits in stream_bot_response.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    large_content = "x" * 3000  # Exceeds Discord's 2000 char limit
    mock_graph.invoke.return_value = {"messages": [AIMessage(content=large_content)]}

    with pytest.raises(RuntimeError, match="Response exceeds Discord message size limit"):
        await message_handler.stream_bot_response(mock_graph, {"messages": [HumanMessage(content="Test input")]})


@pytest.mark.asyncio
async def test_check_for_attachments_total_size_limit(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test total attachment size limit enforcement.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_message = mocker.Mock(spec=discord.Message)
    mock_message.content = ""
    mock_attachment = mocker.Mock(spec=discord.Attachment)
    mock_attachment.size = 60 * 1024 * 1024  # 60MB, exceeds 50MB limit
    mock_message.attachments = [mock_attachment]

    with pytest.raises(RuntimeError, match="Total attachment size .* exceeds .* limit"):
        await message_handler.check_for_attachments(mock_message)


@pytest.mark.asyncio
async def test_check_for_attachments_tenor_gif(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test Tenor GIF URL handling.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_message = mocker.Mock(spec=discord.Message)
    mock_message.content = "https://tenor.com/view/test-gif"
    mock_message.attachments = []

    # Mock the _handle_tenor_gif method
    message_handler._handle_tenor_gif = mocker.AsyncMock(return_value="Processed Tenor GIF")

    result = await message_handler.check_for_attachments(mock_message)
    assert result == "Processed Tenor GIF"
    message_handler._handle_tenor_gif.assert_called_once_with(mock_message, mock_message.content)


@pytest.mark.asyncio
async def test_check_for_attachments_tenor_gif_size_limit(
    message_handler: MessageHandler, mocker: MockerFixture
) -> None:
    """Test Tenor GIF URL size limit enforcement.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_message = mocker.Mock(spec=discord.Message)
    mock_message.content = "https://tenor.com/view/" + "x" * 2049  # Exceeds 2048 limit
    mock_message.attachments = []

    with pytest.raises(ValueError, match="Message content exceeds Discord limit"):
        await message_handler.check_for_attachments(mock_message)


@pytest.mark.asyncio
async def test_check_for_attachments_too_many_image_urls(
    message_handler: MessageHandler, mocker: MockerFixture
) -> None:
    """Test limit on number of image URLs.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_message = mocker.Mock(spec=discord.Message)
    # Create 6 image URLs (exceeds limit of 5)
    mock_message.content = " ".join([f"https://example.com/image{i}.png" for i in range(6)])
    mock_message.attachments = []

    with pytest.raises(ValueError, match="Too many image URLs in message"):
        await message_handler.check_for_attachments(mock_message)


@pytest.mark.asyncio
async def test_check_for_attachments_too_many_attachments(
    message_handler: MessageHandler, mocker: MockerFixture
) -> None:
    """Test limit on number of attachments.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_message = mocker.Mock(spec=discord.Message)
    mock_message.content = ""
    # Create 6 attachments (exceeds limit of 5)
    mock_message.attachments = [mocker.Mock(spec=discord.Attachment) for _ in range(6)]
    for attachment in mock_message.attachments:
        attachment.size = 1024  # Small size to not trigger size limit

    with pytest.raises(ValueError, match="Too many attachments in message"):
        await message_handler.check_for_attachments(mock_message)


@pytest.mark.asyncio
async def test_check_for_attachments_with_discord_attachment(
    message_handler: MessageHandler, mocker: MockerFixture
) -> None:
    """Test Discord attachment handling.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_message = mocker.Mock(spec=discord.Message)
    mock_message.content = ""
    mock_attachment = mocker.Mock(spec=discord.Attachment)
    mock_attachment.size = 1024  # Small size
    mock_message.attachments = [mock_attachment]

    # Mock the _handle_attachment_image method
    message_handler._handle_attachment_image = mocker.AsyncMock(return_value="Processed attachment")

    result = await message_handler.check_for_attachments(mock_message)
    assert result == "Processed attachment"
    message_handler._handle_attachment_image.assert_called_once_with(mock_message)


@pytest.mark.asyncio
async def test_check_for_attachments_error_handling(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test general error handling in check_for_attachments.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_message = mocker.Mock(spec=discord.Message)
    mock_message.content = "test content"
    mock_message.attachments = []

    # Mock _handle_url_image to raise an unexpected error
    message_handler._handle_url_image = mocker.AsyncMock(side_effect=Exception("Unexpected error"))

    # Should return original content on unexpected error
    result = await message_handler.check_for_attachments(mock_message)
    assert result == "test content"


@pytest.mark.asyncio
async def test_stream_bot_response_invalid_format(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test handling of invalid response format.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    mock_graph.invoke.return_value = {"invalid": "format"}

    with pytest.raises(ValueError, match="No response generated"):
        await message_handler.stream_bot_response(mock_graph, {"messages": [HumanMessage(content="Test input")]})


@pytest.mark.asyncio
async def test_stream_bot_response_logging(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test logging in stream_bot_response.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    mock_graph.invoke.return_value = {"messages": [AIMessage(content="Test response")]}

    with structlog.testing.capture_logs() as captured:
        await message_handler.stream_bot_response(mock_graph, {"messages": [HumanMessage(content="Test input")]})

        # Verify no error logs were generated
        assert not any(log.get("level") == "error" for log in captured), "Unexpected error logs found"


@pytest.mark.asyncio
async def test_stream_bot_response_memory_cleanup(message_handler: MessageHandler, mocker: MockerFixture) -> None:
    """Test memory cleanup after response generation.

    Args:
        message_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    mock_graph = mocker.Mock(spec=CompiledStateGraph)
    large_content = "x" * 1000  # Large enough to trigger cleanup
    mock_graph.invoke.return_value = {"messages": [AIMessage(content=large_content)]}

    # Mock gc.collect to track calls
    mock_gc = mocker.patch("gc.collect")

    await message_handler.check_for_attachments(
        mocker.Mock(spec=discord.Message, content=large_content, attachments=[])
    )

    # Verify gc.collect was called
    mock_gc.assert_called()
