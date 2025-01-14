"""Tests for the attachment handler module."""

from __future__ import annotations

import asyncio
import os
import pathlib

from collections.abc import AsyncGenerator, Generator

import aiohttp
import discord

from aioresponses import aioresponses
from discord import Attachment

import pytest

from pytest_mock import MockerFixture

from democracy_exe.chatbot.handlers.attachment_handler import AttachmentHandler
from democracy_exe.constants import MAX_BYTES_UPLOAD_DISCORD, MAX_FILE_UPLOAD_IMAGES_IMGUR


@pytest.fixture
def attachment_handler() -> AttachmentHandler:
    """Create an attachment handler instance.

    Returns:
        An AttachmentHandler instance
    """
    return AttachmentHandler()


@pytest.fixture
def mock_attachment(mocker: MockerFixture) -> Attachment:
    """Create a mock attachment.

    Args:
        mocker: pytest-mock fixture

    Returns:
        A mock Attachment instance
    """
    mock = mocker.Mock(spec=Attachment)
    mock.filename = "test.txt"
    mock.id = "123456"
    mock.proxy_url = "https://example.com/proxy/test.txt"
    mock.size = 1024
    mock.url = "https://example.com/test.txt"
    mock.is_spoiler.return_value = False
    mock.height = None
    mock.width = None
    mock.content_type = "text/plain"
    return mock


def test_attachment_to_dict(attachment_handler: AttachmentHandler, mock_attachment: Attachment) -> None:
    """Test converting attachment to dictionary.

    Args:
        attachment_handler: The attachment handler instance
        mock_attachment: A mock attachment
    """
    result = attachment_handler.attachment_to_dict(mock_attachment)
    assert result["filename"] == "test.txt"
    assert result["id"] == "123456"
    assert result["proxy_url"] == "https://example.com/proxy/test.txt"
    assert result["size"] == 1024
    assert result["url"] == "https://example.com/test.txt"
    assert result["spoiler"] is False
    assert "height" not in result
    assert "width" not in result
    assert result["content_type"] == "text/plain"
    assert result["attachment_obj"] == mock_attachment


def test_attachment_to_dict_with_dimensions(attachment_handler: AttachmentHandler, mock_attachment: Attachment) -> None:
    """Test converting attachment with dimensions to dictionary.

    Args:
        attachment_handler: The attachment handler instance
        mock_attachment: A mock attachment
    """
    mock_attachment.height = 100
    mock_attachment.width = 200
    result = attachment_handler.attachment_to_dict(mock_attachment)
    assert result["height"] == 100
    assert result["width"] == 200


@pytest.mark.asyncio
async def test_download_image(attachment_handler: AttachmentHandler) -> None:
    """Test downloading an image.

    Args:
        attachment_handler: The attachment handler instance
    """
    test_url = "https://example.com/test.jpg"
    test_data = b"test data"

    with aioresponses() as m:
        m.get(test_url, status=200, body=test_data)
        result = await attachment_handler.download_image(test_url)
        assert result is not None
        assert result.getvalue() == test_data


@pytest.mark.asyncio
async def test_download_image_size_limit(attachment_handler: AttachmentHandler) -> None:
    """Test image size limit during download.

    Args:
        attachment_handler: The attachment handler instance
    """
    test_url = "https://example.com/test.jpg"
    test_data = b"test data"
    large_size = MAX_FILE_UPLOAD_IMAGES_IMGUR + 1

    with aioresponses() as m:
        m.get(test_url, status=200, body=test_data, headers={"Content-Length": str(large_size)})
        with pytest.raises(RuntimeError, match=f"Image size {large_size} exceeds {MAX_FILE_UPLOAD_IMAGES_IMGUR} limit"):
            await attachment_handler.download_image(test_url)


@pytest.mark.asyncio
async def test_save_attachment(
    attachment_handler: AttachmentHandler, mock_attachment: Attachment, tmp_path: pathlib.Path
) -> None:
    """Test saving an attachment.

    Args:
        attachment_handler: The attachment handler instance
        mock_attachment: A mock attachment
        tmp_path: pytest temporary path fixture
    """

    async def mock_save(path: str, use_cached: bool = True) -> None:
        with open(path, "w") as f:
            f.write("test content")
        await asyncio.sleep(0.1)

    mock_attachment.save = mock_save
    mock_attachment.size = len("test content")

    await attachment_handler.save_attachment(mock_attachment, str(tmp_path))
    saved_file = tmp_path / "test.txt"
    assert saved_file.exists()
    assert saved_file.read_text() == "test content"


@pytest.mark.asyncio
async def test_save_attachment_invalid_type(
    attachment_handler: AttachmentHandler, mock_attachment: Attachment, tmp_path: pathlib.Path
) -> None:
    """Test saving an attachment with invalid type.

    Args:
        attachment_handler: The attachment handler instance
        mock_attachment: A mock attachment
        tmp_path: pytest temporary path fixture
    """
    mock_attachment.content_type = "application/exe"
    with pytest.raises(ValueError, match="File type application/exe not allowed"):
        await attachment_handler.save_attachment(mock_attachment, str(tmp_path))


@pytest.mark.asyncio
async def test_save_attachment_size_limit(
    attachment_handler: AttachmentHandler, mock_attachment: Attachment, tmp_path: pathlib.Path
) -> None:
    """Test attachment size limit during save.

    Args:
        attachment_handler: The attachment handler instance
        mock_attachment: A mock attachment
        tmp_path: pytest temporary path fixture
    """
    mock_attachment.size = MAX_BYTES_UPLOAD_DISCORD + 1
    with pytest.raises(
        RuntimeError, match=f"Attachment size {mock_attachment.size} exceeds {MAX_BYTES_UPLOAD_DISCORD} limit"
    ):
        await attachment_handler.save_attachment(mock_attachment, str(tmp_path))


@pytest.mark.asyncio
async def test_save_attachment_directory_traversal(
    attachment_handler: AttachmentHandler, mock_attachment: Attachment, tmp_path: pathlib.Path
) -> None:
    """Test directory traversal prevention during save.

    Args:
        attachment_handler: The attachment handler instance
        mock_attachment: A mock attachment
        tmp_path: pytest temporary path fixture
    """
    mock_attachment.filename = "../test.txt"
    mock_attachment.content_type = "text/plain"
    content = "test content"
    mock_attachment.size = len(content)

    async def mock_save(path: str, use_cached: bool = True) -> None:
        with open(path, "w") as f:
            f.write(content)
        await asyncio.sleep(0.1)

    mock_attachment.save = mock_save
    with pytest.raises(ValueError, match="Invalid file path - potential directory traversal attempt"):
        await attachment_handler.save_attachment(mock_attachment, str(tmp_path))


@pytest.mark.asyncio
async def test_save_attachment_verify_saved_file(
    attachment_handler: AttachmentHandler, mock_attachment: Attachment, tmp_path: pathlib.Path
) -> None:
    """Test verification of saved file.

    Args:
        attachment_handler: The attachment handler instance
        mock_attachment: A mock attachment
        tmp_path: pytest temporary path fixture
    """
    content = "test content"
    mock_attachment.size = len(content)

    async def mock_save(path: str, use_cached: bool = True) -> None:
        with open(path, "w") as f:
            f.write(content)
        await asyncio.sleep(0.1)

    mock_attachment.save = mock_save
    await attachment_handler.save_attachment(mock_attachment, str(tmp_path))
    saved_file = tmp_path / "test.txt"
    assert saved_file.exists()
    assert saved_file.read_text() == content


@pytest.mark.asyncio
async def test_handle_save_attachment_locally(
    attachment_handler: AttachmentHandler, mock_attachment: Attachment, tmp_path: pathlib.Path
) -> None:
    """Test saving attachment locally.

    Args:
        attachment_handler: The attachment handler instance
        mock_attachment: A mock attachment
        tmp_path: pytest temporary path fixture
    """
    content = "test content"
    mock_attachment.size = len(content)

    async def mock_save(path: str, use_cached: bool = True) -> None:
        with open(path, "w") as f:
            f.write(content)
        await asyncio.sleep(0.1)

    mock_attachment.save = mock_save
    attm_data = attachment_handler.attachment_to_dict(mock_attachment)
    result = await attachment_handler.handle_save_attachment_locally(attm_data, str(tmp_path))
    assert os.path.exists(result)
    with open(result) as f:
        assert f.read() == content
