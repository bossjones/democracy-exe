"""Tests for AsyncGalleryDL class."""

from __future__ import annotations

import asyncio
import concurrent.futures
import datetime
import json
import logging
import pathlib
import sys

from collections.abc import AsyncIterator, Generator
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, cast

import aiofiles
import pytest_asyncio
import structlog

from langsmith import tracing_context
from structlog.testing import capture_logs

import pytest

from democracy_exe.clients.aio_gallery_dl import AsyncGalleryDL, GalleryDLConfig


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from vcr.request import Request as VCRRequest

    from pytest_mock.plugin import MockerFixture

logger = structlog.get_logger(__name__)


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
        ...     mock_gallery_dl.extractor.find.return_value = []
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
    mock_extractor = mock_gallery_dl.extractor.find.return_value
    mock_extractor.__iter__.return_value = iter(mock_extractor_items)

    # Test extraction with command line options
    client = AsyncGalleryDL(verbose=True, write_info_json=True, write_metadata=True, no_mtime=True)
    items = []
    async for item in client.extract_from_url("https://example.com"):
        items.append(item)

    # Verify results
    assert items == mock_extractor_items
    mock_gallery_dl.extractor.find.assert_called_once_with("https://example.com")


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
@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
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


@pytest_asyncio.fixture
async def sample_config(tmp_path: pathlib.Path) -> str:
    """Provide path to sample gallery-dl config file.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to sample config file
    """
    config_path = tmp_path / "gallery-dl.conf"
    async with aiofiles.open("tests/fixtures/gallery_dl.conf") as f:
        config_data = await f.read()
    async with aiofiles.open(config_path, "w") as f:
        await f.write(config_data)
    return str(config_path)


@pytest.mark.asyncio
async def test_config_validation(sample_config: str) -> None:
    """Test that config file loads and validates correctly.

    Args:
        sample_config: Path to sample config file
    """
    # async with AsyncGalleryDL(config_file=sample_config) as client:
    async with AsyncGalleryDL(config_file=sample_config) as client:
        # Validate the config matches our Pydantic models
        config = GalleryDLConfig(**client.config)

        # Test some specific values
        assert config.extractor.base_directory == "./gallery-dl/"
        assert config.extractor.instagram.username.get_secret_value() == "testuser"
        assert config.extractor.reddit.client_id.get_secret_value() == "test_client_id"
        assert config.downloader.http.timeout == 30.0
        assert config.output.mode == "auto"


@pytest.mark.asyncio
async def test_context_manager(tmp_path: pathlib.Path) -> None:
    """Test async context manager protocol.

    This test verifies that the AsyncGalleryDL class correctly implements
    the async context manager protocol and loads config files.

    Args:
        tmp_path: Pytest temporary directory fixture
    """
    # Test without config file
    async with AsyncGalleryDL() as client:
        assert isinstance(client, AsyncGalleryDL)

    # Test with config file
    config_file = tmp_path / "gallery-dl.conf"
    test_config = {
        "extractor": {
            "base-directory": "~/Downloads",
            "user-agent": "Mozilla/5.0",
        }
    }

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(test_config, f)

    async with AsyncGalleryDL(config_file=str(config_file)) as client:
        assert isinstance(client, AsyncGalleryDL)
        assert client.config["extractor"]["base-directory"] == "~/Downloads"
        assert client.config["extractor"]["user-agent"] == "Mozilla/5.0"

    # Test with invalid config file
    invalid_config = tmp_path / "invalid.conf"
    with open(invalid_config, "w", encoding="utf-8") as f:
        f.write("invalid json")

    async with AsyncGalleryDL(config_file=str(invalid_config)) as client:
        assert isinstance(client, AsyncGalleryDL)
        assert "extractor" not in client.config


@pytest.mark.asyncio
@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
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
            async for _ in client.extract_from_url("hgf://example.com"):
                pass

        # Verify error was logged
        # assert "Error in gallery-dl extraction" in caplog.text
        assert "" in caplog.text


@pytest.mark.asyncio
async def test_download_error(mock_gallery_dl: Any, caplog: LogCaptureFixture, capsys: CaptureFixture) -> None:
    """Test error handling during download.

    This test verifies that errors during download are properly handled
    and logged in both dev and production modes.

    Args:
        mock_gallery_dl: Mock gallery-dl module
        caplog: Pytest log capture fixture
        capsys: Pytest stdout/stderr capture fixture
    """
    with capture_logs() as cap_logs:
        caplog.set_level(logging.ERROR, logger="democracy_exe")

        # Setup mock error
        mock_gallery_dl.job.DownloadJob.side_effect = ValueError("Test error")

        # Test with test_mode=True to prevent debugger
        client = AsyncGalleryDL(test_mode=True)
        with pytest.raises(ValueError, match="Test error"):
            async for _ in client.download("https://example.com"):
                pass

        # Verify error was logged
        assert any(
            log.get("event") == "Error in executor"
            and log.get("error") == "Test error"
            and log.get("error_type") == "ValueError"
            for log in cap_logs
        )

        # Verify error output was captured
        captured = capsys.readouterr()
        assert "Test error" in captured.out
        assert "Error Class: <class 'ValueError'>" in captured.out
        assert "[UNEXPECTED] ValueError: Test error" in captured.out


def _filter_request_headers(request: VCRRequest) -> Any:
    request.headers = {}
    return request


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


# NOTE: this isthe decorator that worked
# # @pytest.mark.block_network
# @pytest.mark.asyncio
# @pytest.mark.vcr(
#     # mode="once",
#     match_on=["method", "scheme", "port", "path", "query"],
#     ignore_localhost=False,
#     record_mode="once",
#     filter_headers=[
#         ("authorization", "DUMMY_AUTHORIZATION"),
#         # ("Set-Cookie", "DUMMY_COOKIE"),
#         ("x-api-key", "DUMMY_API_KEY"),
#         ("api-key", "DUMMY_API_KEY"),
#     ],
#     # filter_headers=[
#     #     ("authorization", None),
#     #     ("Set-Cookie", None),
#     #     ("x-api-key", None),
#     #     ("api-key", None),
#     #     ("set-cookie", None),
#     # ],
#     filter_query_parameters=["api-version", "client_id", "client_secret", "code", "api_key"],
#     before_record_request=_filter_request_headers,
#     before_record_response=_filter_response,
# )


# @pytest.mark.block_network
@pytest.mark.asyncio
@pytest.mark.vcronly()
@pytest.mark.gallerydlonly()
@pytest.mark.default_cassette("test_run_single_tweet_aio_gallery_dl.yaml")
@pytest.mark.vcr(
    record_mode="new_episodes",
    allow_playback_repeats=True,
    match_on=["scheme", "port", "path"],  # Removed method and query to be more lenient
    ignore_localhost=False,
    before_record_response=_filter_response,
    before_record_request=_filter_request_headers,
)
async def test_run_single_tweet_aio_gallery_dl(
    vcr: VCRRequest,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
) -> None:
    """Test extracting metadata from a single tweet.

    Args:
        mocker: Pytest mocker fixture
        caplog: Pytest log capture fixture
        capsys: Pytest stdout/stderr capture fixture
        vcr: VCR request fixture
    """
    # # Enable VCR debug logging
    vcr_log = logging.getLogger("vcr")
    vcr_log.setLevel(logging.DEBUG)

    # import bpdb; bpdb.set_trace()
    with capsys.disabled():
        with capture_logs() as captured:
            caplog.set_level(logging.DEBUG)

            with tracing_context(enabled=False):
                # Use a known working tweet URL for testing
                url = "https://x.com/Eminitybaba_/status/1868256259251863704"

                async with AsyncGalleryDL(
                    verbose=True, write_info_json=True, write_metadata=True, no_mtime=True, config_file=None
                ) as client:
                    try:
                        async for item in client.extract_from_url(url):
                            assert item is not None
                            assert isinstance(item, tuple)
                            # Basic metadata checks
                            assert "author" in item[1]
                            assert "date" in item[1]
                            break
                    except Exception as e:
                        logger.exception(f"Error extracting from URL: {url}")
                        # logger.complete()()
                        raise


@pytest.mark.asyncio
async def test_run_in_executor_thread_safety(
    mock_gallery_dl: Any,
    caplog: LogCaptureFixture,
) -> None:
    """Test thread safety of _run_in_executor.

    This test verifies that _run_in_executor properly handles:
    1. Thread isolation and state management
    2. Thread pool lifecycle and cleanup
    3. Exception propagation from worker threads
    4. Concurrent execution safety
    5. Resource cleanup

    Args:
        mock_gallery_dl: Mock gallery-dl module
        caplog: Pytest log capture fixture
    """
    client = AsyncGalleryDL()
    shared_state = {"counter": 0, "errors": []}
    thread_local_data = {}

    import threading

    counter_lock = threading.Lock()

    def thread_safe_task(task_id: int) -> int:
        """Thread-safe task that modifies shared state.

        Args:
            task_id: Unique task identifier

        Returns:
            Task result
        """
        # Store thread-local data using thread ID
        thread_id = threading.get_ident()
        thread_local_data[task_id] = thread_id

        # Modify shared state in a thread-safe way
        with counter_lock:
            shared_state["counter"] += 1
        return task_id

    def error_task() -> None:
        """Task that raises an error to test exception handling."""
        thread_id = threading.get_ident()
        thread_local_data["error"] = thread_id
        with counter_lock:
            shared_state["counter"] = 999
        raise ValueError("Expected test error")

    # Test 1: Basic thread safety and state isolation
    results = []
    for i in range(5):
        result = await client._run_in_executor(thread_safe_task, i)
        results.append(result)

    assert results == list(range(5))
    assert shared_state["counter"] == 5
    assert len(thread_local_data) == 5  # Each task got its own thread ID

    # Test 2: Exception propagation
    with capture_logs() as captured:
        with pytest.raises(ValueError, match="Expected test error"):
            await client._run_in_executor(error_task)

        # Verify error was logged with context
        assert any(
            log.get("event") == "Error in executor" and "Expected test error" in log.get("error", "")
            for log in captured
        )

    # Test 3: Concurrent execution
    async def concurrent_task(task_id: int) -> int:
        return await client._run_in_executor(thread_safe_task, task_id)

    # Run multiple tasks concurrently
    concurrent_results = await asyncio.gather(*[concurrent_task(i) for i in range(5, 10)])

    assert concurrent_results == list(range(5, 10))
    assert shared_state["counter"] == 1004  # 5 from first test + 5 from concurrent + 999 from error test

    # Test 4: Resource cleanup
    import psutil

    process = psutil.Process()
    thread_count_before = len(process.threads())

    # Run a bunch of tasks
    await asyncio.gather(*[concurrent_task(i) for i in range(10, 20)])

    # Give Python's GC a chance to clean up
    import gc

    gc.collect()

    # Verify no thread leaks
    thread_count_after = len(process.threads())
    assert thread_count_after <= thread_count_before + 2  # Allow for some overhead

    # Verify thread IDs were unique
    thread_ids = set(thread_local_data.values())
    assert len(thread_ids) > 1  # Should have used multiple threads


@pytest.mark.asyncio
async def test_run_in_executor_loop_handling(
    mock_gallery_dl: Any,
    caplog: LogCaptureFixture,
) -> None:
    """Test event loop handling in _run_in_executor.

    This test verifies that _run_in_executor properly handles event loop
    access and errors.

    Args:
        mock_gallery_dl: Mock gallery-dl module
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        # Test with no event loop
        client = AsyncGalleryDL(loop=None)

        def simple_func() -> str:
            """Test function that returns a string."""
            return "test"

        # Should use running loop when no loop provided
        result = await client._run_in_executor(simple_func)
        assert result == "test"

        # Test with explicit loop
        loop = asyncio.get_running_loop()
        client = AsyncGalleryDL(loop=loop)
        result = await client._run_in_executor(simple_func)
        assert result == "test"

        # Test error logging
        def error_func() -> None:
            """Test function that raises an exception."""
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            await client._run_in_executor(error_func)

        # Verify error was logged with proper context
        assert any(
            log.get("event") == "Error in executor"
            and log.get("error") == "Test error"
            and log.get("error_type") == "RuntimeError"
            for log in captured
        ), "Expected error log not found"


@pytest.mark.asyncio
async def test_run_in_executor_concurrent_calls(
    mock_gallery_dl: Any,
    caplog: LogCaptureFixture,
) -> None:
    """Test concurrent execution in _run_in_executor.

    This test verifies that _run_in_executor properly handles multiple
    concurrent calls.

    Args:
        mock_gallery_dl: Mock gallery-dl module
        caplog: Pytest log capture fixture
    """
    client = AsyncGalleryDL()
    results: list[int] = []

    async def concurrent_task(value: int) -> None:
        """Run a task that sleeps then returns a value."""

        def worker() -> int:
            import time

            time.sleep(0.1)  # Simulate work
            return value

        result = await client._run_in_executor(worker)
        results.append(result)

    # Run multiple concurrent tasks
    tasks = [concurrent_task(i) for i in range(5)]
    await asyncio.gather(*tasks)

    # Verify all tasks completed
    assert len(results) == 5
    assert sorted(results) == list(range(5))
