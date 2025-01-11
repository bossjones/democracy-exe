"""Tests for attachment_handler.py functionality."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import pathlib
import tempfile
import uuid

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, List, cast

import aiohttp
import discord
import structlog

from PIL import Image

import pytest

from democracy_exe.chatbot.handlers.attachment_handler import AttachmentHandler


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def attachment_handler() -> AttachmentHandler:
    """Create an AttachmentHandler instance for testing.

    Returns:
        AttachmentHandler: The handler instance for testing
    """
    return AttachmentHandler()


@pytest.fixture
def mock_attachment(mocker: MockerFixture) -> discord.Attachment:
    """Create a mock Discord attachment.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        discord.Attachment: The mock attachment instance
    """
    attachment = mocker.Mock(spec=discord.Attachment)
    attachment.filename = "test.txt"
    attachment.id = "123456789"
    attachment.proxy_url = "https://example.com/proxy/test.txt"
    attachment.size = 1024  # 1KB
    attachment.url = "https://example.com/test.txt"
    attachment.is_spoiler.return_value = False
    attachment.height = None
    attachment.width = None
    attachment.content_type = "text/plain"
    return attachment


@pytest.fixture
def temp_dir() -> str:
    """Create a temporary directory for testing.

    Returns:
        str: Path to the temporary directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.mark.asyncio
async def test_attachment_to_dict(attachment_handler: AttachmentHandler, mock_attachment: discord.Attachment) -> None:
    """Test conversion of attachment to dictionary.

    Args:
        attachment_handler: The handler instance to test
        mock_attachment: The mock attachment instance
    """
    result = attachment_handler.attachment_to_dict(mock_attachment)

    assert result["filename"] == "test.txt"
    assert result["id"] == "123456789"
    assert result["proxy_url"] == "https://example.com/proxy/test.txt"
    assert result["size"] == 1024
    assert result["url"] == "https://example.com/test.txt"
    assert result["spoiler"] is False
    assert "height" not in result
    assert "width" not in result
    assert result["content_type"] == "text/plain"


@pytest.mark.asyncio
async def test_attachment_to_dict_with_dimensions(
    attachment_handler: AttachmentHandler, mock_attachment: discord.Attachment
) -> None:
    """Test conversion of attachment with dimensions to dictionary.

    Args:
        attachment_handler: The handler instance to test
        mock_attachment: The mock attachment instance
    """
    mock_attachment.height = 100
    mock_attachment.width = 200

    result = attachment_handler.attachment_to_dict(mock_attachment)

    assert result["height"] == 100
    assert result["width"] == 200


@pytest.mark.asyncio
async def test_file_to_local_data_dict(attachment_handler: AttachmentHandler, temp_dir: str) -> None:
    """Test conversion of local file to dictionary.

    Args:
        attachment_handler: The handler instance to test
        temp_dir: Path to temporary directory
    """
    # Create test file
    test_file = pathlib.Path(temp_dir) / "test.txt"
    test_file.write_text("test content")

    result = attachment_handler.file_to_local_data_dict(str(test_file), temp_dir)

    assert result["filename"] == f"{temp_dir}/test.txt"
    assert result["size"] == len("test content")
    assert result["ext"] == ".txt"
    assert isinstance(result["api"], pathlib.Path)


@pytest.mark.asyncio
async def test_download_image(attachment_handler: AttachmentHandler, mocker: MockerFixture) -> None:
    """Test image download functionality.

    Args:
        attachment_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    # Mock aiohttp response
    mock_response = mocker.Mock()
    mock_response.status = 200
    mock_response.content_length = 1024
    mock_response.content.iter_chunked.return_value = [b"test data"]

    mock_session = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response

    mocker.patch("aiohttp.ClientSession", return_value=mock_session)

    result = await attachment_handler.download_image("https://example.com/test.jpg")

    assert result is not None
    assert isinstance(result, io.BytesIO)
    assert result.getvalue() == b"test data"


@pytest.mark.asyncio
async def test_download_image_size_limit(attachment_handler: AttachmentHandler, mocker: MockerFixture) -> None:
    """Test image download size limits.

    Args:
        attachment_handler: The handler instance to test
        mocker: Pytest mocker fixture
    """
    # Mock aiohttp response with large content length
    mock_response = mocker.Mock()
    mock_response.status = 200
    mock_response.content_length = 10 * 1024 * 1024  # 10MB

    mock_session = mocker.AsyncMock()
    mock_session.get.return_value.__aenter__.return_value = mock_response

    mocker.patch("aiohttp.ClientSession", return_value=mock_session)

    with pytest.raises(RuntimeError, match="Image size .* exceeds 8MB limit"):
        await attachment_handler.download_image("https://example.com/test.jpg")


@pytest.mark.asyncio
async def test_save_attachment(
    attachment_handler: AttachmentHandler, mock_attachment: discord.Attachment, temp_dir: str
) -> None:
    """Test attachment saving functionality.

    Args:
        attachment_handler: The handler instance to test
        mock_attachment: The mock attachment instance
        temp_dir: Path to temporary directory
    """

    # Mock save method
    async def mock_save(path: pathlib.Path, use_cached: bool = True) -> None:
        path.write_text("test content")

    mock_attachment.save = mock_save

    await attachment_handler.save_attachment(mock_attachment, temp_dir)

    saved_path = pathlib.Path(temp_dir) / "test.txt"
    assert saved_path.exists()
    assert saved_path.read_text() == "test content"


@pytest.mark.asyncio
async def test_save_attachment_invalid_type(
    attachment_handler: AttachmentHandler, mock_attachment: discord.Attachment, temp_dir: str
) -> None:
    """Test saving attachment with invalid file type.

    Args:
        attachment_handler: The handler instance to test
        mock_attachment: The mock attachment instance
        temp_dir: Path to temporary directory
    """
    mock_attachment.content_type = "application/exe"

    with pytest.raises(ValueError, match="File type .* not allowed"):
        await attachment_handler.save_attachment(mock_attachment, temp_dir)


@pytest.mark.asyncio
async def test_save_attachment_size_limit(
    attachment_handler: AttachmentHandler, mock_attachment: discord.Attachment, temp_dir: str
) -> None:
    """Test attachment size limits.

    Args:
        attachment_handler: The handler instance to test
        mock_attachment: The mock attachment instance
        temp_dir: Path to temporary directory
    """
    mock_attachment.size = 10 * 1024 * 1024  # 10MB

    with pytest.raises(RuntimeError, match="Attachment size .* exceeds 8MB limit"):
        await attachment_handler.save_attachment(mock_attachment, temp_dir)


@pytest.mark.asyncio
async def test_save_attachment_directory_traversal(
    attachment_handler: AttachmentHandler, mock_attachment: discord.Attachment, temp_dir: str
) -> None:
    """Test protection against directory traversal.

    Args:
        attachment_handler: The handler instance to test
        mock_attachment: The mock attachment instance
        temp_dir: Path to temporary directory
    """
    mock_attachment.filename = "../test.txt"

    with pytest.raises(ValueError, match="Invalid file path"):
        await attachment_handler.save_attachment(mock_attachment, temp_dir)


@pytest.mark.asyncio
async def test_save_attachment_verify_saved_file(
    attachment_handler: AttachmentHandler, mock_attachment: discord.Attachment, temp_dir: str
) -> None:
    """Test verification of saved files.

    Args:
        attachment_handler: The handler instance to test
        mock_attachment: The mock attachment instance
        temp_dir: Path to temporary directory
    """

    # Mock save method to create file with wrong size
    async def mock_save(path: pathlib.Path, use_cached: bool = True) -> None:
        path.write_text("wrong size content")

    mock_attachment.save = mock_save
    mock_attachment.size = 5  # Different from actual content size

    with pytest.raises(RuntimeError, match="Saved file size does not match attachment size"):
        await attachment_handler.save_attachment(mock_attachment, temp_dir)

    # Verify file was cleaned up
    saved_path = pathlib.Path(temp_dir) / "test.txt"
    assert not saved_path.exists()


@pytest.mark.asyncio
async def test_handle_save_attachment_locally(
    attachment_handler: AttachmentHandler, mock_attachment: discord.Attachment, temp_dir: str
) -> None:
    """Test local attachment saving.

    Args:
        attachment_handler: The handler instance to test
        mock_attachment: The mock attachment instance
        temp_dir: Path to temporary directory
    """
    attm_data = attachment_handler.attachment_to_dict(mock_attachment)

    # Mock save method
    async def mock_save(path: str, use_cached: bool = True) -> None:
        pathlib.Path(path).write_text("test content")

    mock_attachment.save = mock_save

    result = await attachment_handler.handle_save_attachment_locally(attm_data, temp_dir)

    assert result.startswith(temp_dir)
    assert pathlib.Path(result).exists()
    assert pathlib.Path(result).read_text() == "test content"
