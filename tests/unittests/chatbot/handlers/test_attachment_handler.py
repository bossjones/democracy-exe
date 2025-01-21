# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
"""Unit tests for the AttachmentHandler class."""

from __future__ import annotations

import asyncio
import base64
import io
import pathlib

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast

import aiohttp
import aioresponses
import discord
import structlog

from discord import Attachment, File, HTTPException, Message


logger = structlog.get_logger(__name__)

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
        AttachmentHandler: A new instance of AttachmentHandler
    """
    return AttachmentHandler()


@pytest.fixture
def mock_attachment(mocker: MockerFixture) -> Attachment:
    """Create a mock Discord Attachment for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Attachment: A mocked Discord Attachment object
    """
    mock_attm = mocker.Mock(spec=Attachment)
    mock_attm.filename = "test.png"
    mock_attm.id = 123456789
    mock_attm.proxy_url = "https://proxy.url/test.png"
    mock_attm.size = 1024
    mock_attm.url = "https://cdn.discord.com/test.png"
    mock_attm.is_spoiler.return_value = False
    mock_attm.height = 100
    mock_attm.width = 200
    mock_attm.content_type = "image/png"
    return mock_attm


@pytest.fixture
async def mock_aioresponse() -> aioresponses.aioresponses:
    """Create a mock aioresponse for testing.

    Returns:
        aioresponses.aioresponses: A mock aioresponse instance
    """
    with aioresponses.aioresponses() as m:
        yield m


class TestAttachmentHandler:
    """Test suite for AttachmentHandler class."""

    def test_attachment_to_dict(self, attachment_handler: AttachmentHandler, mock_attachment: Attachment) -> None:
        """Test converting an attachment to a dictionary.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_attachment: A mock Discord Attachment
        """
        result = attachment_handler.attachment_to_dict(mock_attachment)

        assert isinstance(result, dict)
        assert result["filename"] == "test.png"
        assert result["id"] == 123456789
        assert result["proxy_url"] == "https://proxy.url/test.png"
        assert result["size"] == 1024
        assert result["url"] == "https://cdn.discord.com/test.png"
        assert result["spoiler"] is False
        assert result["height"] == 100
        assert result["width"] == 200
        assert result["content_type"] == "image/png"
        assert result["attachment_obj"] == mock_attachment

    def test_file_to_local_data_dict(self, attachment_handler: AttachmentHandler, tmp_path: pathlib.Path) -> None:
        """Test converting a local file to a metadata dictionary.

        Args:
            attachment_handler: The AttachmentHandler instance
            tmp_path: Pytest temporary directory fixture
        """
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = attachment_handler.file_to_local_data_dict(str(test_file), "test_dir")

        assert isinstance(result, dict)
        assert result["filename"] == "test_dir/test.txt"
        assert result["size"] > 0
        assert result["ext"] == ".txt"
        assert isinstance(result["api"], pathlib.Path)

    @pytest.mark.asyncio
    async def test_download_image_success(
        self,
        attachment_handler: AttachmentHandler,
        mock_aioresponse: aioresponses.aioresponses,
    ) -> None:
        """Test successful image download.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_aioresponse: Mock aioresponse fixture
        """
        test_image_data = b"fake_image_data"
        test_url = "https://example.com/test.png"

        mock_aioresponse.get(
            test_url, status=200, body=test_image_data, headers={"Content-Length": str(len(test_image_data))}
        )

        result = await attachment_handler.download_image(test_url)
        assert result is not None
        assert isinstance(result, io.BytesIO)
        assert result.getvalue() == test_image_data

    @pytest.mark.asyncio
    async def test_download_image_404(
        self,
        attachment_handler: AttachmentHandler,
        mock_aioresponse: aioresponses.aioresponses,
    ) -> None:
        """Test image download with 404 response.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_aioresponse: Mock aioresponse fixture
        """
        test_url = "https://example.com/nonexistent.png"
        mock_aioresponse.get(test_url, status=404)

        result = await attachment_handler.download_image(test_url)
        assert result is None

    @pytest.mark.asyncio
    async def test_download_image_too_large(
        self,
        attachment_handler: AttachmentHandler,
        mock_aioresponse: aioresponses.aioresponses,
    ) -> None:
        """Test image download with file too large.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_aioresponse: Mock aioresponse fixture
        """
        test_url = "https://example.com/too_large.png"
        mock_aioresponse.get(
            test_url, status=200, headers={"Content-Length": str(attachment_handler._max_image_size + 1)}
        )

        with pytest.raises(RuntimeError, match="Image size .* exceeds .* limit"):
            await attachment_handler.download_image(test_url)

    @pytest.mark.asyncio
    async def test_download_image_timeout(
        self,
        attachment_handler: AttachmentHandler,
        mock_aioresponse: aioresponses.aioresponses,
    ) -> None:
        """Test image download with timeout.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_aioresponse: Mock aioresponse fixture
        """
        test_url = "https://example.com/timeout.png"
        mock_aioresponse.get(test_url, exception=TimeoutError())

        with pytest.raises(RuntimeError, match="Image download timed out"):
            await attachment_handler.download_image(test_url)

    @pytest.mark.asyncio
    async def test_download_image_client_error(
        self,
        attachment_handler: AttachmentHandler,
        mock_aioresponse: aioresponses.aioresponses,
    ) -> None:
        """Test image download with client error.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_aioresponse: Mock aioresponse fixture
        """
        test_url = "https://example.com/error.png"
        mock_aioresponse.get(test_url, exception=aiohttp.ClientError("Connection failed"))

        with pytest.raises(aiohttp.ClientError, match="Connection failed"):
            await attachment_handler.download_image(test_url)

    @pytest.mark.asyncio
    async def test_file_to_data_uri(self, attachment_handler: AttachmentHandler, mocker: MockerFixture) -> None:
        """Test converting a Discord File to a data URI.

        Args:
            attachment_handler: The AttachmentHandler instance
            mocker: Pytest mocker fixture
        """
        # Create a mock File object
        mock_fp = io.BytesIO(b"test_data")
        mock_file = mocker.Mock(spec=File)
        mock_file.fp = mock_fp
        mock_file.fp.readable = mocker.Mock(return_value=True)

        result = await attachment_handler.file_to_data_uri(mock_file)

        assert isinstance(result, str)
        assert result.startswith("data:image;base64,")
        # Verify the base64 encoded content
        base64_content = result.split(",")[1]
        decoded_content = base64.b64decode(base64_content)
        assert decoded_content == b"test_data"

    @pytest.mark.asyncio
    async def test_data_uri_to_file(self, attachment_handler: AttachmentHandler) -> None:
        """Test converting a data URI to a Discord File.

        Args:
            attachment_handler: The AttachmentHandler instance
        """
        test_data = b"test_data"
        base64_data = base64.b64encode(test_data).decode("ascii")
        data_uri = f"data:image;base64,{base64_data}"

        result = await attachment_handler.data_uri_to_file(data_uri, "test.png")

        assert isinstance(result, File)
        assert result.filename == "test.png"
        assert not result.spoiler

    def test_path_for(self, attachment_handler: AttachmentHandler, mock_attachment: Attachment) -> None:
        """Test generating a path for an attachment.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_attachment: A mock Discord Attachment
        """
        result = attachment_handler.path_for(mock_attachment, "./test_dir")

        assert isinstance(result, pathlib.Path)
        assert str(result.relative_to(result.parent)) == "test.png"
        assert result.parent.name == "test_dir"

    @pytest.mark.asyncio
    async def test_save_attachment(
        self,
        attachment_handler: AttachmentHandler,
        mock_attachment: Attachment,
        tmp_path: pathlib.Path,
        mocker: MockerFixture,
    ) -> None:
        """Test saving an attachment to disk.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_attachment: A mock Discord Attachment
            tmp_path: Pytest temporary directory fixture
            mocker: Pytest mocker fixture
        """
        # Configure mock attachment
        mock_attachment.content_type = "image/png"
        mock_attachment.size = len("test content")

        # Mock the save method to actually create a file
        async def mock_save(path: str, use_cached: bool = True) -> None:
            with open(path, "w") as f:
                f.write("test content")
            await asyncio.sleep(0.1)  # Small delay to ensure file is written

        mock_attachment.save = mock_save

        # Save the attachment
        await attachment_handler.save_attachment(mock_attachment, str(tmp_path))

        # Verify file was saved
        save_path = tmp_path / "test.png"
        assert save_path.exists()
        assert save_path.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_handle_save_attachment_locally(
        self,
        attachment_handler: AttachmentHandler,
        mock_attachment: Attachment,
        tmp_path: pathlib.Path,
        mocker: MockerFixture,
    ) -> None:
        """Test saving an attachment locally.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_attachment: A mock Discord Attachment
            tmp_path: Pytest temporary directory fixture
        """
        attm_data = {"id": "123456789", "filename": "test.png", "attachment_obj": mock_attachment}

        # Mock the save method
        mock_attachment.save = mocker.AsyncMock()

        result = await attachment_handler.handle_save_attachment_locally(attm_data, str(tmp_path))

        assert isinstance(result, str)
        assert result == f"{tmp_path}/orig_123456789_test.png"
        mock_attachment.save.assert_called_once()

    def test_get_attachments(
        self, attachment_handler: AttachmentHandler, mock_attachment: Attachment, mocker: MockerFixture
    ) -> None:
        """Test retrieving attachments from a Discord message.

        Args:
            attachment_handler: The AttachmentHandler instance
            mock_attachment: A mock Discord Attachment
            mocker: Pytest mocker fixture
        """
        # Create a mock Message
        mock_message = mocker.Mock(spec=Message)
        mock_message.attachments = [mock_attachment]

        result = attachment_handler.get_attachments(mock_message)

        assert isinstance(result, tuple)
        assert len(result) == 4
        attachment_data_list, local_files, local_data, media_files = result

        assert len(attachment_data_list) == 1
        assert isinstance(attachment_data_list[0], dict)
        assert attachment_data_list[0]["filename"] == "test.png"

        assert isinstance(local_files, list)
        assert isinstance(local_data, list)
        assert isinstance(media_files, list)
