from __future__ import annotations

import json
import logging

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Dict

import structlog

from structlog.testing import LogCapture

import pytest

from democracy_exe.bot_logger.logsetup import StructLogHandler, configure_logger, custom_logger, get_custom_logger


if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


@pytest.fixture(name="log_output")
def fixture_log_output() -> LogCapture:
    """Create a LogCapture fixture for testing structlog output.

    Returns:
        LogCapture: A structlog LogCapture instance
    """
    return LogCapture()


def test_custom_logger_exists() -> None:
    """Test that the custom logger is properly initialized."""
    logger = get_custom_logger()
    assert logger is not None
    assert isinstance(logger, structlog.stdlib.BoundLogger)


def test_struct_log_handler() -> None:
    """Test the StructLogHandler class with different log levels."""
    handler = StructLogHandler()
    logger = logging.getLogger("test_handler")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    with structlog.testing.capture_logs() as captured:
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")

    assert len(captured) == 5
    assert captured[0]["event"] == "debug message"
    assert captured[1]["event"] == "info message"
    assert captured[2]["event"] == "warning message"
    assert captured[3]["event"] == "error message"
    assert captured[4]["event"] == "critical message"


@pytest.mark.parametrize("enable_json_logs", [True, False])
def test_configure_logger(enable_json_logs: bool, caplog: LogCaptureFixture) -> None:
    """Test the configure_logger function with different output formats.

    Args:
        enable_json_logs: Whether to enable JSON logging
        caplog: pytest's log capture fixture
    """
    configure_logger(enable_json_logs=enable_json_logs)
    logger = structlog.get_logger()

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


@pytest.mark.asyncio
async def test_async_logging(log_output: LogCapture) -> None:
    """Test async logging capabilities.

    Args:
        log_output: The LogCapture fixture
    """
    configure_logger(enable_json_logs=False)
    logger = structlog.get_logger()

    test_message = "async test message"
    await logger.ainfo(test_message)

    assert len(log_output.entries) == 1
    assert log_output.entries[0]["event"] == test_message


def test_logging_with_context(log_output: LogCapture) -> None:
    """Test logging with bound context information."""
    configure_logger(enable_json_logs=False)
    logger = structlog.get_logger()

    context = {"user": "test_user", "action": "test_action"}
    bound_logger = logger.bind(**context)
    test_message = "context test message"

    bound_logger.info(test_message)

    assert len(log_output.entries) == 1
    log_entry = log_output.entries[0]
    assert log_entry["event"] == test_message
    assert log_entry["user"] == context["user"]
    assert log_entry["action"] == context["action"]


def test_logging_with_extra_processors(log_output: LogCapture) -> None:
    """Test logging with additional processors."""

    def add_test_info(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        """Add test information to the log event.

        Args:
            logger: The logger instance
            method_name: The logging method name
            event_dict: The current event dictionary

        Returns:
            Dict[str, Any]: Updated event dictionary
        """
        event_dict["test_info"] = "test_value"
        return event_dict

    structlog.configure(
        processors=[add_test_info, log_output],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )

    logger = structlog.get_logger()
    test_message = "processor test message"

    logger.info(test_message)

    assert len(log_output.entries) == 1
    log_entry = log_output.entries[0]
    assert log_entry["event"] == test_message
    assert log_entry["test_info"] == "test_value"
