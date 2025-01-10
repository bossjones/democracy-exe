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

import datetime
import random
import string
import sys


# from datetime import datetime
from io import BytesIO

import aiofiles
import aiohttp
import structlog


logger = structlog.get_logger(__name__)

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


MALFORMED_TOKEN = "asdf"
INVALID_TOKEN = "z" * 62

# Need bytes type for Python3
DUMMY_PAYLOAD = string.ascii_letters.encode("ascii")

RANDOM_FOLDER = random.sample(string.ascii_letters, 15)
TIMESTAMP = str(datetime.datetime.now(datetime.UTC))
STATIC_FILE = "/test.txt"


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
    mock.read = mocker.AsyncMock(return_value=b"test content")
    mock.release = mocker.AsyncMock()
    mock.__aenter__ = mocker.AsyncMock(return_value=mock)
    mock.__aexit__ = mocker.AsyncMock()
    return mock


@pytest.fixture
async def dbx_client(mock_session: MockerFixture) -> AsyncGenerator[AsyncDropboxAPI, None]:
    """Create a test Dropbox client.

    Args:
        mock_session: Mocked aiohttp session

    Yields:
        AsyncDropboxAPI: Test client instance
    """
    client = AsyncDropboxAPI()
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


# def _filter_response(response: VCRRequest) -> VCRRequest:
#     """Filter response before recording.

#     Args:
#         response: The response to filter

#     Returns:
#         The filtered response
#     """
#     if "retry-after" in response["headers"]:
#         response["headers"]["retry-after"] = "0"  # type: ignore
#     return response


def _filter_response(response: VCRRequest) -> VCRRequest:
    """
    Filter the response before recording.

    If the response has a 'retry-after' header, we set it to 0 to avoid waiting for the retry time.

    Args:
        response (VCRRequest): The response to filter.

    Returns:
        VCRRequest: The filtered response.
    """

    if "retry-after" in response["headers"]:
        response["headers"]["retry-after"] = "0"  # type: ignore
    if "x-stainless-arch" in response["headers"]:
        response["headers"]["x-stainless-arch"] = "arm64"  # type: ignore

    if "apim-request-id" in response["headers"]:
        response["headers"]["apim-request-id"] = ["9a705e27-2f04-4bd6-abd8-01848165ebbf"]  # type: ignore

    if "azureml-model-session" in response["headers"]:
        response["headers"]["azureml-model-session"] = ["d089-20240815073451"]  # type: ignore

    if "x-ms-client-request-id" in response["headers"]:
        response["headers"]["x-ms-client-request-id"] = ["9a705e27-2f04-4bd6-abd8-01848165ebbf"]  # type: ignore

    if "x-ratelimit-remaining-requests" in response["headers"]:
        response["headers"]["x-ratelimit-remaining-requests"] = ["144"]  # type: ignore
    if "x-ratelimit-remaining-tokens" in response["headers"]:
        response["headers"]["x-ratelimit-remaining-tokens"] = ["143324"]  # type: ignore
    if "x-request-id" in response["headers"]:
        response["headers"]["x-request-id"] = ["143324"]  # type: ignore
    if "Set-Cookie" in response["headers"]:
        response["headers"]["Set-Cookie"] = [  # type: ignore
            "__cf_bm=fake;path=/; expires=Tue, 15-Oct-24 23:22:45 GMT; domain=.api.openai.com; HttpOnly;Secure; SameSite=None",
            "_cfuvid=fake;path=/; domain=.api.openai.com; HttpOnly; Secure; SameSite=None",
        ]  # type: ignore
    if "set-cookie" in response["headers"]:
        response["headers"]["set-cookie"] = [  # type: ignore
            "guest_id_marketing=v1%3FAKEBROTHER; Max-Age=63072000; Expires=Sat, 19 Dec 2026 19:52:20 GMT; Path=/; Domain=.x.com; Secure; SameSite=None",
            "guest_id_ads=v1%3FAKEBROTHER; Max-Age=63072000; Expires=Sat, 19 Dec 2026 19:52:20 GMT; Path=/; Domain=.x.com; Secure; SameSite=None",
            "personalization_id=v1_SUPERFAKE; Max-Age=63072000; Expires=Sat, 19 Dec 2026 19:52:20 GMT; Path=/; Domain=.x.com; Secure; SameSite=None",
            "guest_id=v1%3FAKEBROTHER; Max-Age=63072000; Expires=Sat, 19 Dec 2026 19:52:20 GMT; Path=/; Domain=.x.com; Secure; SameSite=None",
        ]  # type: ignore

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
async def test_dropbox_api_validate(
    mock_response: MockerFixture,
    mocker: MockerFixture,
) -> None:
    """Test Dropbox API token validation.

    Args:
        mock_response: Mocked response fixture
        mocker: Pytest mocker fixture
    """

    # Mock the response to return matching nonce
    async def mock_do_request(*args, **kwargs):
        data = json.loads(kwargs.get("data", "{}"))
        return {"result": data.get("query")}

    client = AsyncDropboxAPI()
    client._do_request = mocker.AsyncMock(side_effect=mock_do_request)

    assert await client.validate() is True
    await client._cleanup()


@pytest.mark.asyncio
async def test_dropbox_api_download_file(
    mock_response: MockerFixture,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Test file download from Dropbox.

    Args:
        mock_response: Mocked response
        tmp_path: Pytest temporary path fixture
        mocker: Pytest mocker fixture
    """
    test_content = b"test file content"
    mock_response.read = mocker.AsyncMock(return_value=test_content)
    mock_response.content_type = "application/octet-stream"
    local_path = tmp_path / "downloaded.txt"

    client = AsyncDropboxAPI()
    client._do_request = mocker.AsyncMock(return_value=test_content)

    result = await client.download_file("/test.txt", str(local_path))
    assert result == str(local_path)
    assert local_path.read_bytes() == test_content
    await client._cleanup()


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.dropboxonly()
@pytest.mark.default_cassette("test_dropbox_api_download_file.yaml")
async def test_dropbox_api_download_file_improved(
    tmp_path: pathlib.Path,
    dbx_client: AsyncDropboxAPI,
    mock_response: MockerFixture,
    mocker: MockerFixture,
) -> None:
    """Test file download from Dropbox.

    Args:

        tmp_path: Pytest temporary path fixture
    """

    # @pytest_asyncio.fixture
    # async def mock_tweet_data_with_media(mock_tweet_data: dict[str, Any], tmp_path: pathlib.Path) -> dict[str, Any]:
    #     """Create mock tweet data with media files for testing.

    #     Args:
    #         mock_tweet_data: Base mock tweet data
    #         tmp_path: Temporary directory path

    #     Returns:
    #         Mock tweet data with media files
    #     """

    random_filename = "".join(RANDOM_FOLDER)
    random_path = f"/Test/{TIMESTAMP}/{random_filename}"
    test_contents = DUMMY_PAYLOAD
    # Create a temporary file
    random_dir = tmp_path / random_filename
    random_dir.mkdir(parents=True, exist_ok=True)
    local_path = random_dir / "downloaded.txt"
    # Setup mock response and session
    # Setup mock response
    mock_response.read = mocker.AsyncMock(return_value=test_contents)
    mock_response.content_type = "application/octet-stream"
    mock_response.status = 200

    # Setup mock session
    mock_session = mocker.MagicMock()
    mock_session.request = mocker.AsyncMock()
    mock_session.request.return_value.__aenter__.return_value = mock_response

    # Mock _do_request to return the actual bytes content
    async def mock_do_request(*args, **kwargs):
        return await mock_response.read()

    dbx_client._do_request = mock_do_request

    # Download file
    result = await dbx_client.download_file(random_path, str(local_path))
    assert result == str(local_path)
    assert local_path.read_bytes() == test_contents


@pytest.mark.asyncio
async def test_dropbox_api_upload_single(
    mock_response: MockerFixture,
    tmp_path: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Test single file upload to Dropbox.

    Args:
        mock_response: Mocked response
        tmp_path: Pytest temporary path fixture
        mocker: Pytest mocker fixture
    """
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    response_data = {"path_display": "/test.txt"}
    client = AsyncDropboxAPI()
    client._do_request = mocker.AsyncMock(return_value=response_data)

    result = await client.upload_single(str(test_file), "/test.txt")
    assert result["path_display"] == "/test.txt"
    await client._cleanup()


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
        mocker: Pytest mocker fixture
    """
    # Mock the _do_request method directly
    dbx_client._do_request = mocker.AsyncMock(return_value={"url": "https://dropbox.com/s/test"})

    result = await dbx_client.create_shared_link("/test.txt")
    assert result == "https://dropbox.com/s/test"


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
        mocker: Pytest mocker fixture
        tmp_path: Pytest temporary path fixture
    """
    # Create a test file larger than the chunk size
    test_file = tmp_path / "large_test.txt"
    test_file.write_bytes(b"x" * (5 * 1024 * 1024))  # 5MB file

    # Mock _do_request to return appropriate responses for each stage
    do_request_mock = mocker.AsyncMock()
    # Create a list of responses for each chunk plus start/finish/link
    responses = []
    responses.append({"session_id": "test-session"})  # Start session

    # Add responses for each chunk (5MB file with 1MB chunks = 5 chunks)
    chunk_count = 5
    for _ in range(chunk_count - 1):  # -1 because last chunk is handled by finish
        responses.append({})  # Append chunk responses

    responses.append({"path_display": "/large_test.txt"})  # Finish upload
    responses.append({"url": "https://dropbox.com/s/test"})  # Create shared link

    do_request_mock.side_effect = responses
    dbx_client._do_request = do_request_mock

    progress_callback = mocker.AsyncMock()
    result = await dbx_client.dropbox_upload(
        str(test_file),
        "/large_test.txt",
        chunk_size=1024 * 1024,  # 1MB chunks
        progress_callback=progress_callback,
    )

    assert result == "https://dropbox.com/s/test"
    assert progress_callback.call_count > 0

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
async def test_request_rate_limit_handling(mocker: MockerFixture) -> None:
    """Test Request class rate limit handling.

    Args:
        mocker: Pytest mocker fixture
    """
    # Create mock responses
    rate_limited_response = mocker.AsyncMock()
    rate_limited_response.status = 429
    rate_limited_response.ok = False

    success_response = mocker.AsyncMock()
    success_response.status = 200
    success_response.ok = True

    # Create mock request function
    mock_func = mocker.AsyncMock(
        side_effect=[
            rate_limited_response,  # First call returns rate limit
            success_response,  # Second call succeeds
        ]
    )

    # Create and execute request
    request = Request(mock_func, "test_url", {}, retry_count=2)
    response = await request

    # Verify behavior
    assert response == success_response
    assert mock_func.call_count == 2


@pytest.mark.asyncio
async def test_safe_file_handler_thread_safety(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler thread safety.

    Tests concurrent access patterns and proper locking behavior.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "concurrent.txt"
    write_count = 5
    results: list[bool] = []
    errors: list[Exception] = []

    async def write_file(content: str) -> None:
        try:
            async with SafeFileHandler(test_file, "a") as f:
                await f.write(content + "\n")
                await asyncio.sleep(0.1)  # Simulate work
            results.append(True)
        except Exception as e:
            errors.append(e)
            results.append(False)

    # Run concurrent writes
    tasks = [write_file(f"line {i}") for i in range(write_count)]
    await asyncio.gather(*tasks)

    # Verify results
    assert len(errors) == 0, f"Encountered errors: {errors}"
    assert len(results) == write_count
    assert all(results)

    # Verify file contents
    async with SafeFileHandler(test_file, "r") as f:
        content = await f.read()
    assert len(content.splitlines()) == write_count


@pytest.mark.asyncio
async def test_safe_file_handler_cleanup_on_error(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler cleanup behavior on errors.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "cleanup_test.txt"

    # Test cleanup on write error
    with pytest.raises(OSError):
        async with SafeFileHandler(tmp_path / "nonexistent" / "test.txt", "w") as f:
            await f.write("test")
    assert not test_file.exists()

    # Test cleanup on runtime error
    with pytest.raises(RuntimeError):
        async with SafeFileHandler(test_file, "w") as f:
            await f.write("test")
            raise RuntimeError("Test error")
    assert not test_file.exists()

    # Test no cleanup when cleanup_on_error=False
    with pytest.raises(RuntimeError):
        async with SafeFileHandler(test_file, "w", cleanup_on_error=False) as f:
            await f.write("test")
            raise RuntimeError("Test error")
    assert test_file.exists()
    assert test_file.read_text() == "test"


@pytest.mark.asyncio
async def test_safe_file_handler_state_management(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler state management.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "state_test.txt"
    handler = SafeFileHandler(test_file, "w")

    # Test reuse prevention
    async with handler:
        await handler.file.write("test")  # type: ignore

    assert handler._closed

    # Attempt to reuse
    with pytest.raises(RuntimeError, match="Cannot reuse closed SafeFileHandler"):
        async with handler:
            pass

    # Test explicit close
    handler2 = SafeFileHandler(test_file, "w")
    await handler2.close()
    assert handler2._closed
    assert handler2.file is None


@pytest.mark.asyncio
async def test_safe_file_handler_directory_creation(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler directory creation behavior.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    nested_path = tmp_path / "deep" / "nested" / "dir" / "test.txt"

    # Test automatic directory creation
    async with SafeFileHandler(nested_path, "w") as f:
        await f.write("test")

    assert nested_path.exists()
    assert nested_path.read_text() == "test"

    # Test directory creation failure
    readonly_path = tmp_path / "readonly"
    readonly_path.mkdir()
    os.chmod(readonly_path, 0o444)  # Make readonly

    if os.access(readonly_path, os.W_OK):
        pytest.skip("Cannot create readonly directory for testing")

    with pytest.raises(OSError):
        async with SafeFileHandler(readonly_path / "subdir" / "test.txt", "w"):
            pass


@pytest.mark.asyncio
async def test_safe_file_handler_event_loop_safety() -> None:
    """Test SafeFileHandler event loop safety."""
    # Test creation outside async context
    with pytest.raises(RuntimeError, match="must be created from an async context"):

        def create_handler() -> None:
            SafeFileHandler("test.txt", "r")

        create_handler()

    # Test event loop reference clearing
    handler = SafeFileHandler("test.txt", "r")
    assert handler._loop is not None
    await handler.close()
    assert handler._loop is None


@pytest.mark.asyncio
async def test_safe_file_handler_error_propagation(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler error handling and propagation.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "error_test.txt"

    # Test OSError propagation
    with pytest.raises(OSError) as exc_info:
        async with SafeFileHandler(tmp_path / "nonexistent" / "test.txt", "r"):
            pass
    assert "File operation failed" in str(exc_info.value)

    # Test error during write
    class WriteError(Exception):
        pass

    with pytest.raises(WriteError):
        async with SafeFileHandler(test_file, "w") as f:
            await f.write("start")
            raise WriteError("Test error during write")

    # Verify cleanup occurred
    assert not test_file.exists()


@pytest.mark.asyncio
async def test_safe_file_handler_modes(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler with different file modes.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "modes_test.txt"

    # Test write mode
    async with SafeFileHandler(test_file, "w") as f:
        await f.write("test")
    assert test_file.read_text() == "test"

    # Test append mode
    async with SafeFileHandler(test_file, "a") as f:
        await f.write("_append")
    assert test_file.read_text() == "test_append"

    # Test read mode
    async with SafeFileHandler(test_file, "r") as f:
        content = await f.read()
    assert content == "test_append"

    # Test binary mode
    binary_data = b"binary\x00data"
    async with SafeFileHandler(test_file, "wb") as f:
        await f.write(binary_data)
    async with SafeFileHandler(test_file, "rb") as f:
        content = await f.read()
    assert content == binary_data


@pytest.mark.asyncio
async def test_safe_file_handler_concurrent_reads(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler concurrent read operations.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "concurrent_read.txt"
    test_file.write_text("test content")

    read_count = 5
    results: list[str] = []
    errors: list[Exception] = []

    async def read_file() -> None:
        try:
            async with SafeFileHandler(test_file, "r") as f:
                content = await f.read()
                await asyncio.sleep(0.1)  # Simulate work
                results.append(content)
        except Exception as e:
            errors.append(e)

    # Run concurrent reads
    tasks = [read_file() for _ in range(read_count)]
    await asyncio.gather(*tasks)

    # Verify results
    assert len(errors) == 0, f"Encountered errors: {errors}"
    assert len(results) == read_count
    assert all(content == "test content" for content in results)


@pytest.mark.asyncio
async def test_safe_file_handler_resource_cleanup(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler resource cleanup in various scenarios.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file = tmp_path / "cleanup_test.txt"

    # Test normal cleanup
    handler = SafeFileHandler(test_file, "w")
    async with handler:
        await handler.file.write("test")  # type: ignore
    assert handler._closed
    assert handler.file is None

    # Test cleanup after error
    handler = SafeFileHandler(test_file, "w")
    try:
        async with handler:
            await handler.file.write("test")  # type: ignore
            raise RuntimeError("Test error")
    except RuntimeError:
        pass
    assert handler._closed
    assert handler.file is None

    # Test cleanup with explicit close
    handler = SafeFileHandler(test_file, "w")
    await handler.close()
    assert handler._closed
    assert handler.file is None

    # Test double close
    await handler.close()  # Should not raise
    assert handler._closed


@pytest.mark.asyncio
async def test_safe_file_handler_context_nesting(tmp_path: pathlib.Path) -> None:
    """Test SafeFileHandler nested context behavior.

    Args:
        tmp_path: Pytest temporary path fixture
    """
    test_file1 = tmp_path / "test1.txt"
    test_file2 = tmp_path / "test2.txt"

    # Test nested handlers
    async with SafeFileHandler(test_file1, "w") as f1:
        await f1.write("file1")
        async with SafeFileHandler(test_file2, "w") as f2:
            await f2.write("file2")

    assert test_file1.read_text() == "file1"
    assert test_file2.read_text() == "file2"

    # Test error in nested context
    with pytest.raises(RuntimeError):
        async with SafeFileHandler(test_file1, "w") as f1:
            await f1.write("new1")
            async with SafeFileHandler(test_file2, "w") as f2:
                await f2.write("new2")
                raise RuntimeError("Test error")

    # Verify both files were cleaned up
    assert not test_file1.exists()
    assert not test_file2.exists()
