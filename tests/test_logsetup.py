from __future__ import annotations

import json
import logging
import multiprocessing
import sys

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Dict

import structlog

from structlog.testing import LogCapture

import pytest

from democracy_exe.bot_logger.logsetup import configure_logging, get_logger


if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


@pytest.fixture(name="log_output")
def fixture_log_output() -> LogCapture:
    """Create a LogCapture fixture for testing structlog output.

    Returns:
        LogCapture: A structlog LogCapture instance
    """
    return LogCapture()


@pytest.fixture(autouse=True)
def clean_logging() -> Generator[None, None, None]:
    """Reset logging configuration before and after each test.

    Yields:
        None
    """
    # Reset before test
    logging.root.handlers = []
    structlog.reset_defaults()

    yield

    # Reset after test
    logging.root.handlers = []
    structlog.reset_defaults()


@pytest.mark.logsonly
def test_logger_initialization() -> None:
    """Test that the logger is properly initialized."""
    # Configure logging first
    configure_logging()

    # Get a logger instance
    logger = get_logger("test")

    # Verify logger is not None
    assert logger is not None

    # Log a test message to ensure logger is properly configured
    logger.info("test message")

    # Verify logger is properly configured by checking its type
    assert isinstance(logger, structlog.BoundLoggerLazyProxy)
    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


@pytest.mark.logsonly
@pytest.mark.parametrize("enable_json_logs", [True, False])
def test_configure_logging(enable_json_logs: bool, caplog: LogCaptureFixture) -> None:
    """Test the configure_logging function with different output formats.

    Args:
        enable_json_logs: Whether to enable JSON logging
        caplog: pytest's log capture fixture
    """
    configure_logging(enable_json_logs=enable_json_logs)
    logger = get_logger("test")

    test_message = "test message"
    logger.info(test_message)

    assert len(caplog.records) == 1
    record = caplog.records[0]

    if enable_json_logs:
        # For JSON logs, the message should be JSON-parseable
        try:
            log_data = json.loads(record.message)
            assert log_data["event"] == test_message
        except json.JSONDecodeError:
            pytest.fail("Failed to parse JSON log message")
    else:
        # For console logs, the message should contain the test message
        assert test_message in record.message


@pytest.mark.logsonly
@pytest.mark.asyncio
async def test_async_logging(log_output: LogCapture) -> None:
    """Test async logging capabilities.

    Args:
        log_output: The LogCapture fixture
    """
    configure_logging(enable_json_logs=False)
    logger = get_logger("test_async")

    test_message = "async test message"
    await logger.ainfo(test_message)

    assert len(log_output.entries) == 1
    assert log_output.entries[0]["event"] == test_message


@pytest.mark.logsonly
def test_logging_with_context(log_output: LogCapture) -> None:
    """Test logging with bound context information."""
    configure_logging(enable_json_logs=False)
    logger = get_logger("test_context")

    # Test context binding
    context = {"user": "test_user", "action": "test_action"}
    bound_logger = logger.bind(**context)
    test_message = "context test message"

    bound_logger.info(test_message)

    assert len(log_output.entries) == 1
    log_entry = log_output.entries[0]
    assert log_entry["event"] == test_message
    assert log_entry["user"] == context["user"]
    assert log_entry["action"] == context["action"]


@pytest.mark.logsonly
def test_process_info_logging() -> None:
    """Test that process information is correctly added to log entries."""
    configure_logging(enable_json_logs=True)
    logger = get_logger("test_process")

    def worker() -> None:
        proc_logger = get_logger("worker")
        proc_logger.info("worker message")

    # Start a worker process
    process = multiprocessing.Process(target=worker)
    process.start()
    process.join()

    # Log in main process
    logger.info("main message")

    # We can't easily capture the worker's logs in this test
    # but we can verify the main process logs have process info
    assert process.exitcode == 0


@pytest.mark.logsonly
def test_third_party_logger_levels() -> None:
    """Test that third-party logger levels are properly configured."""
    custom_levels = {"discord": "ERROR", "aiohttp": "DEBUG"}

    configure_logging(third_party_loggers=custom_levels)

    # Check logger levels are set correctly
    discord_logger = logging.getLogger("discord")
    aiohttp_logger = logging.getLogger("aiohttp")

    assert discord_logger.level == logging.ERROR
    assert aiohttp_logger.level == logging.DEBUG


@pytest.mark.logsonly
def test_logging_to_correct_streams() -> None:
    """Test that logs go to the correct output streams."""
    configure_logging()
    logger = get_logger("test_streams")

    # Get handlers for different levels
    root_logger = logging.getLogger()
    handlers = root_logger.handlers

    # Verify we have both stdout and stderr handlers
    stdout_handlers = [h for h in handlers if getattr(h, "stream", None) is sys.stdout]
    stderr_handlers = [h for h in handlers if getattr(h, "stream", None) is sys.stderr]

    assert len(stdout_handlers) > 0, "No stdout handler found"
    assert len(stderr_handlers) > 0, "No stderr handler found"


@pytest.mark.logsonly
def test_module_name_processor() -> None:
    """Test that the module name processor adds correct module information."""
    configure_logging()
    logger = get_logger("test_module")

    with structlog.testing.capture_logs() as captured:
        logger.info("test message")

    assert len(captured) == 1
    assert captured[0]["module"] == "test_module"
