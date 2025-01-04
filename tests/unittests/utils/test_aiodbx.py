"""Tests for democracy_exe.utils.aiodbx."""

# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pylint: disable=unused-import
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
# pyright: reportUninitializedInstanceVariable=false
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from langsmith import tracing_context
from pydantic import SecretStr


# Type imports must come before any usage of the types
if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.logging import LogCaptureFixture
    from vcr.request import Request as VCRRequest

import aiofiles
import aiohttp

from loguru import logger

import pytest

from pytest_mock import MockerFixture

from democracy_exe import aio_settings
from democracy_exe.aio_settings import aiosettings
from democracy_exe.utils.aiodbx import (
    AsyncDropboxAPI,
    DropboxAPIError,
    Request,
    SafeFileHandler,
    aio_path_basename,
    aio_path_exists,
    aio_path_getsize,
    retry_with_backoff,
    safe_aiofiles_open,
)


@pytest.fixture
def mock_session(mocker: MockerFixture) -> Generator[MockerFixture, None, None]:
    """Create a mock aiohttp ClientSession.

    Returns:
        MagicMock: Mocked session
    """
    with mocker.patch("aiohttp.ClientSession") as mock:
        yield mock


@pytest.fixture
def mock_response(mocker: MockerFixture) -> MockerFixture:
    """Create a mock aiohttp ClientResponse.

    Returns:
        MagicMock: Mocked response
    """
    mock = mocker.MagicMock()
    mock.status = 200
    mock.ok = True
    mock.content_type = "application/json"
    mock.json = mocker.AsyncMock(return_value={"result": "success"})
    mock.text = mocker.AsyncMock(return_value="response text")
    mock.release = mocker.AsyncMock()
    return mock


@pytest.fixture
async def dbx_client(mock_session: MockerFixture) -> AsyncGenerator[AsyncDropboxAPI, None]:
    """Create a test Dropbox client.

    Args:
        mock_session: Mocked aiohttp session

    Yields:
        AsyncDropboxAPI: Test client instance
    """
    client = AsyncDropboxAPI(access_token="test_token")  # noqa: S106
    yield client
    await client._cleanup()


@pytest.mark.asyncio
async def test_aio_path_exists(tmp_path: pathlib.Path) -> None:
    """Test aio_path_exists function.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "test.txt"
    test_file.touch()

    assert await aio_path_exists(test_file) is True
    assert await aio_path_exists(tmp_path / "nonexistent.txt") is False


@pytest.mark.asyncio
async def test_aio_path_basename() -> None:
    """Test aio_path_basename function."""
    assert await aio_path_basename("/path/to/file.txt") == "file.txt"
    assert await aio_path_basename("file.txt") == "file.txt"


@pytest.mark.asyncio
async def test_aio_path_getsize(tmp_path: pathlib.Path) -> None:
    """Test aio_path_getsize function.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    size = await aio_path_getsize(test_file)
    assert size == len("test content")


def _filter_request_headers(request: VCRRequest) -> Any:
    """Filter request headers before recording.

    Args:
        request: The request to filter

    Returns:
        The filtered request
    """
    request.headers = {}
    return request


def _filter_response(response: VCRRequest) -> VCRRequest:
    """Filter response before recording.

    Args:
        response: The response to filter

    Returns:
        The filtered response
    """
    if "retry-after" in response["headers"]:
        response["headers"]["retry-after"] = "0"  # type: ignore
    return response


async def test_safe_file_handler(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler context manager.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "test.txt"

    async with SafeFileHandler(test_file, "w") as f:
        await f.write("test content")

    assert test_file.read_text() == "test content"

    # Test cleanup on error
    with pytest.raises(OSError):
        async with SafeFileHandler(tmp_path / "nonexistent/test.txt", "w"):
            pass


@pytest.mark.asyncio
async def test_retry_with_backoff(mocker: MockerFixture) -> None:
    """Test retry_with_backoff decorator."""
    mock_func = mocker.AsyncMock()
    mock_func.side_effect = [ValueError("error"), ValueError("error"), "success"]

    result = await retry_with_backoff(mock_func, max_retries=2, initial_delay=0.1)
    assert result == "success"
    assert mock_func.call_count == 3

    # Test max retries exceeded
    mock_func.reset_mock()
    mock_func.side_effect = ValueError("error")
    with pytest.raises(ValueError):
        await retry_with_backoff(mock_func, max_retries=2, initial_delay=0.1)


@pytest.mark.asyncio
@pytest.mark.vcronly()
@pytest.mark.dropboxonly()
@pytest.mark.default_cassette("test_dropbox_api_validate.yaml")
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query"],
    ignore_localhost=False,
    before_record_response=_filter_response,
    before_record_request=_filter_request_headers,
)
async def test_dropbox_api_validate(
    vcr: VCRRequest,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
) -> None:
    """Test Dropbox API token validation.

    Args:
        vcr: VCR request fixture
        caplog: Log capture fixture
        capsys: Capture sys output fixture
    """
    # Enable VCR debug logging
    vcr_log = logging.getLogger("vcr")
    vcr_log.setLevel(logging.DEBUG)

    with tracing_context(enabled=False):
        # Create client with real token from env
        # token = os.getenv("DROPBOX_TOKEN")
        token_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_token)
        # app_key_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_app_key)
        # app_secret_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_app_secret)
        # refresh_token_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_token)

        token: str = token_secret.get_secret_value()

        if not token:
            pytest.skip("DROPBOX_TOKEN environment variable not set")

        async with AsyncDropboxAPI(token) as client:
            try:
                assert await client.validate() is True
            except Exception as e:
                logger.exception("Error validating Dropbox token")
                logger.complete()
                raise


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_dropbox_api_download_file(
    dbx_client: AsyncDropboxAPI,
    mock_response: MockerFixture,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Test file download from Dropbox.

    Args:
        dbx_client: Test Dropbox client
        mock_response: Mocked response
        tmp_path: Pytest temporary path fixture
    """
    test_content = b"test file content"
    mock_response.text = mocker.AsyncMock(return_value=test_content)
    mock_response.content_type = "application/octet-stream"
    local_path = tmp_path / "downloaded.txt"

    result = await dbx_client.download_file("/test.txt", str(local_path))
    assert result == str(local_path)
    assert local_path.read_bytes() == test_content


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_dropbox_api_upload_single(
    dbx_client: AsyncDropboxAPI,
    mock_response: MockerFixture,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Test single file upload to Dropbox.

    Args:
        dbx_client: Test Dropbox client
        mock_response: Mocked response
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    mock_response.json = mocker.AsyncMock(return_value={"path_display": "/test.txt"})
    mock_response.status = 200
    mock_response.ok = True

    result = await dbx_client.upload_single(str(test_file), "/test.txt")
    assert (await result.json())["path_display"] == "/test.txt"


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_dropbox_api_create_shared_link(
    dbx_client: AsyncDropboxAPI,
    mock_response: MockerFixture,
    mocker: MockerFixture,
) -> None:
    """Test creating shared link for Dropbox file.

    Args:
        dbx_client: Test Dropbox client
        mock_response: Mocked response
    """
    mock_response.json = mocker.AsyncMock(return_value={"url": "https://dropbox.com/s/test"})
    mock_response.status = 200
    mock_response.ok = True

    result = await dbx_client.create_shared_link("/test.txt")
    assert (await result.json())["url"] == "https://dropbox.com/s/test"


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_dropbox_api_get_connection_metrics(dbx_client: AsyncDropboxAPI) -> None:
    """Test getting connection metrics.

    Args:
        dbx_client: Test Dropbox client
    """
    metrics = await dbx_client.get_connection_metrics()
    assert "active_connections" in metrics
    assert "acquired_connections" in metrics
    assert "connection_limit" in metrics
    assert "connection_timeouts" in metrics


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_dropbox_api_chunked_upload(
    dbx_client: AsyncDropboxAPI, mock_response: MockerFixture, mocker: MockerFixture, tmp_path: pathlib.Path
) -> None:
    """Test chunked file upload to Dropbox.

    Args:
        dbx_client: Test Dropbox client
        mock_response: Mocked response
        tmp_path: Pytest temporary path fixture
    """
    # Create a test file larger than the chunk size
    test_file = tmp_path / "large_test.txt"
    test_file.write_bytes(b"x" * (5 * 1024 * 1024))  # 5MB file

    # Mock responses for each stage of chunked upload
    mock_response.json = mocker.AsyncMock()
    mock_response.json.side_effect = [
        {"session_id": "test-session"},  # Start session
        {},  # Append chunk
        {"path_display": "/large_test.txt"},  # Finish upload
        {"url": "https://dropbox.com/s/test"},  # Create shared link
    ]
    mock_response.status = 200
    mock_response.ok = True
    mock_response.content_type = "application/json"

    progress_callback = mocker.AsyncMock()
    result = await dbx_client.dropbox_upload(
        str(test_file),
        "/large_test.txt",
        chunk_size=1024 * 1024,  # 1MB chunks
        progress_callback=progress_callback,
    )

    assert result == "https://dropbox.com/s/test"
    assert progress_callback.call_count > 0


@pytest.mark.asyncio
async def test_request_retry_logic(mocker: MockerFixture, mock_response: MockerFixture) -> None:
    """Test Request class retry logic.

    Args:
        mock_response: Mocked response
    """
    mock_func = mocker.AsyncMock(
        side_effect=[
            aiohttp.ClientError(),  # First attempt fails
            mock_response,  # Second attempt succeeds
        ]
    )

    request = Request(mock_func, "test_url", {}, retry_count=2)
    response = await request

    assert response == mock_response
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_request_rate_limit_handling(mocker: MockerFixture, mock_response: MockerFixture) -> None:
    """Test Request class rate limit handling.

    Args:
        mock_response: Mocked response
    """
    rate_limited_response = mock_response
    rate_limited_response.status = 429

    mock_func = mocker.AsyncMock(
        side_effect=[
            rate_limited_response,  # Rate limited
            mock_response,  # Success after retry
        ]
    )

    request = Request(mock_func, "test_url", {}, retry_count=2)
    response = await request

    assert response == mock_response
    assert mock_func.call_count == 2
