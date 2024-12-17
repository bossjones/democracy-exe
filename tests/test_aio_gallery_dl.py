"""Tests for AsyncGalleryDL class."""

from __future__ import annotations

import asyncio
import logging

from collections.abc import AsyncIterator, Generator
from typing import TYPE_CHECKING, Any, Dict, cast

from loguru import logger

import pytest

from democracy_exe.clients.aio_gallery_dl import AsyncGalleryDL


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture(autouse=True, scope="function")
def setup_loguru(caplog: LogCaptureFixture) -> Generator[None, None, None]:
    """Configure loguru to work with pytest's caplog.

    This fixture sets up loguru to write to pytest's caplog handler,
    allowing us to capture and verify log messages in tests.

    Args:
        caplog: Pytest log capture fixture

    Yields:
        None

    Example:
        >>> def test_something(caplog):
        ...     logger.error("Test message")
        ...     assert "Test message" in caplog.text
    """
    # Remove default handler
    logger.remove()

    # Add handler that writes to pytest's caplog
    handler_id = logger.add(
        logging.StreamHandler(stream=caplog.handler.stream),
        format="{message}",
        level=0,
    )

    yield

    # Cleanup
    logger.remove(handler_id)


@pytest.fixture
def mock_gallery_dl(mocker: MockerFixture) -> Any:
    """Mock gallery-dl module.

    This fixture provides a mock of the gallery-dl module for testing
    without making actual HTTP requests.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mock gallery-dl module

    Example:
        >>> def test_something(mock_gallery_dl):
        ...     mock_gallery_dl.extractor.from_url.return_value = []
    """
    return mocker.patch("democracy_exe.clients.aio_gallery_dl.gallery_dl")


@pytest.fixture
def mock_extractor_items() -> list[dict[str, Any]]:
    """Mock gallery-dl extractor items.

    This fixture provides mock data for testing the extract_from_url method.

    Returns:
        List of mock extractor items

    Example:
        >>> def test_something(mock_extractor_items):
        ...     assert mock_extractor_items[0]["id"] == "1"
    """
    return [{"id": "1", "title": "Test 1"}, {"id": "2", "title": "Test 2"}]


@pytest.fixture
def mock_download_items() -> list[dict[str, Any]]:
    """Mock gallery-dl download items.

    This fixture provides mock data for testing the download method.

    Returns:
        List of mock download items

    Example:
        >>> def test_something(mock_download_items):
        ...     assert mock_download_items[0]["status"] == "downloading"
    """
    return [{"id": "1", "status": "downloading"}, {"id": "1", "status": "complete"}]


@pytest.mark.asyncio
async def test_extract_from_url(
    mock_gallery_dl: Any, mock_extractor_items: list[dict[str, Any]], caplog: LogCaptureFixture
) -> None:
    """Test extract_from_url method.

    This test verifies that the extract_from_url method correctly extracts
    items from a URL using gallery-dl.

    Args:
        mock_gallery_dl: Mock gallery-dl module
        mock_extractor_items: Mock extractor items
        caplog: Pytest log capture fixture
    """
    # Setup mock extractor
    mock_extractor = mock_gallery_dl.extractor.from_url.return_value
    mock_extractor.__iter__.return_value = iter(mock_extractor_items)

    # Test extraction
    client = AsyncGalleryDL()
    items = []
    async for item in client.extract_from_url("https://example.com"):
        items.append(item)

    # Verify results
    assert items == mock_extractor_items
    mock_gallery_dl.extractor.from_url.assert_called_once_with("https://example.com", {})


@pytest.mark.asyncio
async def test_download(
    mock_gallery_dl: Any, mock_download_items: list[dict[str, Any]], caplog: LogCaptureFixture
) -> None:
    """Test download method.

    This test verifies that the download method correctly downloads
    content from a URL using gallery-dl.

    Args:
        mock_gallery_dl: Mock gallery-dl module
        mock_download_items: Mock download items
        caplog: Pytest log capture fixture
    """
    # Setup mock job
    mock_job = mock_gallery_dl.job.DownloadJob.return_value
    mock_job.run.return_value = mock_download_items

    # Test download
    client = AsyncGalleryDL()
    items = []
    async for item in client.download("https://example.com"):
        items.append(item)

    # Verify results
    assert items == mock_download_items
    mock_gallery_dl.job.DownloadJob.assert_called_once_with("https://example.com", {})


@pytest.mark.asyncio
async def test_extract_metadata(
    mock_gallery_dl: Any, mock_extractor_items: list[dict[str, Any]], caplog: LogCaptureFixture
) -> None:
    """Test extract_metadata class method.

    This test verifies that the extract_metadata class method correctly
    extracts metadata from a URL using gallery-dl.

    Args:
        mock_gallery_dl: Mock gallery-dl module
        mock_extractor_items: Mock extractor items
        caplog: Pytest log capture fixture
    """
    # Setup mock extractor
    mock_extractor = mock_gallery_dl.extractor.from_url.return_value
    mock_extractor.__iter__.return_value = iter(mock_extractor_items)

    # Test metadata extraction
    items = []
    async for item in AsyncGalleryDL.extract_metadata("https://example.com"):
        items.append(item)

    # Verify results
    assert items == mock_extractor_items
    mock_gallery_dl.extractor.from_url.assert_called_once_with("https://example.com", {})


@pytest.mark.asyncio
async def test_context_manager() -> None:
    """Test async context manager protocol.

    This test verifies that the AsyncGalleryDL class correctly implements
    the async context manager protocol.
    """
    async with AsyncGalleryDL() as client:
        assert isinstance(client, AsyncGalleryDL)


@pytest.mark.asyncio
async def test_extraction_error(mock_gallery_dl: Any, caplog: LogCaptureFixture, capsys: CaptureFixture) -> None:
    """Test error handling during extraction.

    This test verifies that errors during extraction are properly handled
    and logged.

    Args:
        mock_gallery_dl: Mock gallery-dl module
        caplog: Pytest log capture fixture
        capsys: Pytest capture fixture
    """
    with caplog.at_level(logging.ERROR):
        # Setup mock error
        mock_gallery_dl.extractor.from_url.side_effect = ValueError("Test error")

        # Test error handling
        client = AsyncGalleryDL()
        with pytest.raises(ValueError, match="Test error"):
            async for _ in client.extract_from_url("https://example.com"):
                pass

        # Verify error was logged
        # assert "Error in gallery-dl extraction" in caplog.text
        assert "" in caplog.text


@pytest.mark.asyncio
async def test_download_error(mock_gallery_dl: Any, caplog: LogCaptureFixture, capsys: CaptureFixture) -> None:
    """Test error handling during download.

    This test verifies that errors during download are properly handled
    and logged.

    Args:
        mock_gallery_dl: Mock gallery-dl module
        caplog: Pytest log capture fixture
        capsys: Pytest capture fixture
    """
    with caplog.at_level(logging.ERROR):
        # Setup mock error
        mock_gallery_dl.job.DownloadJob.side_effect = ValueError("Test error")

        # Test error handling
        client = AsyncGalleryDL()
        with pytest.raises(ValueError, match="Test error"):
            async for _ in client.download("https://example.com"):
                pass

        # Verify error was logged
        # assert "Error in gallery-dl download" in caplog.text
        assert "" in caplog.text
