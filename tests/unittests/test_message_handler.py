# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Tests for the message handler module."""

from __future__ import annotations

import asyncio
import io

from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import structlog

import pytest

from democracy_exe.chatbot.handlers.message_handler import MessageHandler


@pytest.fixture
def mock_bot() -> MagicMock:
    """Create a mock bot instance.

    Returns:
        MagicMock: Mock bot instance
    """
    return MagicMock()


@pytest.fixture
def message_handler(mock_bot: MagicMock) -> MessageHandler:
    """Create a message handler instance.

    Args:
        mock_bot: Mock bot instance

    Returns:
        MessageHandler: Message handler instance
    """
    return MessageHandler(mock_bot)


@pytest.fixture
def mock_message() -> MagicMock:
    """Create a mock message instance.

    Returns:
        MagicMock: Mock message instance
    """
    message = MagicMock(spec=discord.Message)
    message.content = "Test message"
    message.attachments = []
    return message


@pytest.mark.asyncio
async def test_check_for_attachments_basic(message_handler: MessageHandler, mock_message: MagicMock) -> None:
    """Test basic message processing without attachments.

    Args:
        message_handler: Message handler instance
        mock_message: Mock message instance
    """
    result = await message_handler.check_for_attachments(mock_message)
    assert result == "Test message"


@pytest.mark.asyncio
async def test_check_for_attachments_size_limit(message_handler: MessageHandler, mock_message: MagicMock) -> None:
    """Test attachment size limit enforcement.

    Args:
        message_handler: Message handler instance
        mock_message: Mock message instance
    """
    attachment = MagicMock(spec=discord.Attachment)
    attachment.size = message_handler._max_total_size + 1
    mock_message.attachments = [attachment]

    with pytest.raises(RuntimeError, match="Total attachment size .* exceeds .* limit"):
        await message_handler.check_for_attachments(mock_message)


@pytest.mark.asyncio
async def test_check_for_attachments_task_tracking(message_handler: MessageHandler, mock_message: MagicMock) -> None:
    """Test task tracking during message processing.

    Args:
        message_handler: Message handler instance
        mock_message: Mock message instance
    """
    with structlog.testing.capture_logs() as captured:
        result = await message_handler.check_for_attachments(mock_message)
        assert result == "Test message"
        assert any(log.get("event") == "Resource cleanup completed" for log in captured), "Task tracking log not found"


@pytest.mark.asyncio
async def test_stream_bot_response_basic(message_handler: MessageHandler) -> None:
    """Test basic bot response streaming.

    Args:
        message_handler: Message handler instance
    """
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"messages": ["Test response"]})

    result = await message_handler.stream_bot_response(mock_graph, {})
    assert result == "Test response"


@pytest.mark.asyncio
async def test_stream_bot_response_size_limit(message_handler: MessageHandler) -> None:
    """Test response size limit enforcement.

    Args:
        message_handler: Message handler instance
    """
    mock_graph = MagicMock()
    large_response = "x" * (message_handler._resource_manager._limits.max_response_size_mb * 1024 * 1024 + 1)
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [large_response]})

    with pytest.raises(RuntimeError, match="Response exceeds .* size limit"):
        await message_handler.stream_bot_response(mock_graph, {})


@pytest.mark.asyncio
async def test_stream_bot_response_timeout(message_handler: MessageHandler) -> None:
    """Test response timeout handling.

    Args:
        message_handler: Message handler instance
    """
    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(side_effect=TimeoutError())

    with pytest.raises(RuntimeError, match="Response generation timed out"):
        await message_handler.stream_bot_response(mock_graph, {})


@pytest.mark.asyncio
async def test_handle_url_image(message_handler: MessageHandler) -> None:
    """Test URL image handling.

    Args:
        message_handler: Message handler instance
    """
    test_url = "http://example.com/test.jpg"
    test_data = b"test image data"
    mock_response = io.BytesIO(test_data)
    mock_download = AsyncMock(return_value=mock_response)

    with patch.object(message_handler.attachment_handler, "download_image", mock_download):
        result = await message_handler.handle_url_image(test_url)
        assert result == test_url
        mock_download.assert_called_once_with(test_url)


@pytest.mark.asyncio
async def test_handle_url_image_size_limit(message_handler: MessageHandler) -> None:
    """Test URL image size limit enforcement.

    Args:
        message_handler: Message handler instance
    """
    test_url = "http://example.com/test.jpg"
    test_data = b"x" * (message_handler._max_image_size + 1)
    mock_response = io.BytesIO(test_data)
    mock_download = AsyncMock(return_value=mock_response)

    with patch.object(message_handler.attachment_handler, "download_image", mock_download):
        with pytest.raises(RuntimeError, match="Image size .* exceeds .* limit"):
            await message_handler.handle_url_image(test_url)


@pytest.mark.asyncio
async def test_handle_attachment_image(message_handler: MessageHandler) -> None:
    """Test attachment image handling.

    Args:
        message_handler: Message handler instance
    """
    test_data = b"test image data"
    mock_response = io.BytesIO(test_data)
    mock_download = AsyncMock(return_value=mock_response)

    mock_attachment = MagicMock(spec=discord.Attachment)
    mock_attachment.size = len(test_data)
    mock_attachment.url = "http://example.com/test.jpg"

    with patch.object(message_handler.attachment_handler, "download_image", mock_download):
        result = await message_handler.handle_attachment_image(mock_attachment)
        assert result == mock_attachment.url
        mock_download.assert_called_once_with(mock_attachment.url)


@pytest.mark.asyncio
async def test_resource_cleanup(message_handler: MessageHandler, mock_message: MagicMock) -> None:
    """Test resource cleanup after message processing.

    Args:
        message_handler: Message handler instance
        mock_message: Mock message instance
    """
    with structlog.testing.capture_logs() as captured:
        await message_handler.check_for_attachments(mock_message)
        assert any(log.get("event") == "Resource cleanup completed" for log in captured), (
            "Resource cleanup log not found"
        )


@pytest.mark.asyncio
async def test_concurrent_downloads(message_handler: MessageHandler) -> None:
    """Test concurrent download handling.

    Args:
        message_handler: Message handler instance
    """
    test_url = "http://example.com/test.jpg"
    test_data = b"test image data"
    mock_response = io.BytesIO(test_data)
    mock_download = AsyncMock(return_value=mock_response)

    with patch.object(message_handler.attachment_handler, "download_image", mock_download):
        tasks = [asyncio.create_task(message_handler.handle_url_image(test_url)) for _ in range(10)]
        await asyncio.gather(*tasks)
        assert mock_download.call_count == 10
