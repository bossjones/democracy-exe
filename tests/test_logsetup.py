from __future__ import annotations

import datetime
import json
import logging
import multiprocessing
import sys

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Dict

import structlog

from structlog.testing import LogCapture

import pytest

from democracy_exe.bot_logger.logsetup import _add_module_name, _add_process_info, configure_logging, get_logger


if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


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

    # Log a test message and verify output
    with structlog.testing.capture_logs() as captured:
        test_message = "test message"
        logger.info(test_message)

        # Verify log output
        assert len(captured) == 1, "Expected exactly one log message"
        assert captured[0]["event"] == test_message, "Incorrect message content"

        # Verify logger is properly configured
        assert hasattr(logger, "info"), "Logger missing info method"
        assert hasattr(logger, "debug"), "Logger missing debug method"
        assert hasattr(logger, "warning"), "Logger missing warning method"
        assert hasattr(logger, "error"), "Logger missing error method"
        assert all(hasattr(logger, attr) for attr in ["info", "debug", "warning", "error"])


@pytest.mark.logsonly
@pytest.mark.parametrize("enable_json_logs", [True, False])
def test_configure_logging(enable_json_logs: bool) -> None:
    """Test the configure_logging function with different output formats.

    Args:
        enable_json_logs: Whether to enable JSON logging
    """
    configure_logging(enable_json_logs=enable_json_logs)
    logger = get_logger("test")

    with structlog.testing.capture_logs() as captured:
        test_message = "test message"
        logger.info(test_message)

        assert len(captured) == 1, "Expected exactly one log message"
        log_entry = captured[0]
        assert log_entry["event"] == test_message, "Incorrect message content"
        assert log_entry["log_level"] == "info", "Incorrect log level"


@pytest.mark.logsonly
@pytest.mark.asyncio
async def test_async_logging() -> None:
    """Test async logging capabilities."""
    configure_logging(enable_json_logs=False)
    logger = get_logger("test_async")

    test_message = "async test message"

    with structlog.testing.capture_logs() as captured:
        logger.info(test_message)  # Use regular info() since we're using BoundLogger

        assert len(captured) == 1, "Expected exactly one log message"
        assert captured[0]["event"] == test_message, "Incorrect message content"
        assert captured[0]["log_level"] == "info", "Incorrect log level"


@pytest.mark.logsonly
def test_logging_with_context() -> None:
    """Test logging with bound context information."""
    configure_logging(enable_json_logs=False)
    logger = get_logger("test_context")

    # Test context binding
    context = {"user": "test_user", "action": "test_action"}

    with structlog.testing.capture_logs() as captured:
        bound_logger = logger.bind(**context)
        test_message = "context test message"
        bound_logger.info(test_message)

        assert len(captured) == 1, "Expected exactly one log message"
        log_entry = captured[0]
        assert log_entry["event"] == test_message, "Incorrect message content"
        assert log_entry["user"] == context["user"], "Missing or incorrect user context"
        assert log_entry["action"] == context["action"], "Missing or incorrect action context"


@pytest.mark.logsonly
def test_process_info_logging() -> None:
    """Test that process information is correctly added to log entries."""
    # Test the processor directly first
    event_dict = _add_process_info("test_logger", "info", {"event": "main message"})

    # Verify processor adds the expected fields
    assert "process_name" in event_dict, "Processor failed to add process name"
    assert "process_id" in event_dict, "Processor failed to add process ID"
    assert isinstance(event_dict["process_id"], int), "Process ID should be an integer"

    # Now test through the logging system
    configure_logging(enable_json_logs=False)
    logger = get_logger("test_process")

    with structlog.testing.capture_logs() as captured:
        logger.info("main message")

        assert len(captured) == 1, "Expected exactly one log message"
        log_entry = captured[0]
        assert log_entry["event"] == "main message", "Incorrect message content"


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
    """Test that the module name processor adds correct module information.

    This test verifies both:
    1. Direct processor functionality
    2. Integration with the logging system
    """
    # Test the processor function directly
    test_event = {"event": "test message"}
    processed_event = _add_module_name("test_module", "info", test_event)

    assert "module" in processed_event, "Processor failed to add module field"
    assert processed_event["module"] == "test_module", "Processor added incorrect module name"
    assert processed_event["event"] == "test message", "Processor modified event message"

    # Test the processor integrated in the logging system
    configure_logging()
    logger = get_logger("test_module")

    with structlog.testing.capture_logs() as captured:
        test_message = "integration test message"
        logger.info(test_message)

        assert len(captured) == 1, "Expected exactly one log message"
        log_entry = captured[0]

        # Verify logger name is preserved
        assert log_entry["logger"] == "test_module", "Missing or incorrect logger name"

        # Verify module name is added
        assert log_entry["module"] == "test_module", "Missing or incorrect module name"

        # Verify original message is preserved
        assert log_entry["event"] == test_message, "Log message was modified"
