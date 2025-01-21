# pyright: reportUnusedFunction=false
# pyright: reportUndefinedVariable=false
# pyright: reportInvalidTypeForm=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import datetime
import json
import logging
import multiprocessing
import sys

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Dict, cast

import pytest_structlog
import structlog

from freezegun import freeze_time
from pytest_structlog import StructuredLogCapture
from structlog.processors import TimeStamper, add_log_level
from structlog.stdlib import BoundLogger
from structlog.testing import CapturingLogger, LogCapture
from structlog.typing import BindableLogger, EventDict, Processor, WrappedLogger

import pytest

from democracy_exe.bot_logger.logsetup import configure_logging, get_logger


logger = structlog.get_logger("test_logsetup")


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture(autouse=True, scope="function")
def set_logging_environment(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("LOGGING_ENVIRONMENT", "testing")


# @pytest.fixture(name="log")
# def fixture_log():
#     return LogCapture()


# @pytest.fixture(autouse=True, scope="function")
# def fixture_configure_structlog_simple(log: pytest_structlog.StructuredLogCapture) -> Generator[None, None, None]:
#     structlog.configure(processors=[log])
#     yield
#     structlog.reset_defaults()


@pytest.fixture
def stdlib_bound_logger_configure(log: pytest_structlog.StructuredLogCapture) -> None:
    """Configure structlog with stdlib and LogCapture for testing.

    Args:
        log: The LogCapture fixture
    """
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.format_exc_info,
            log,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,
    )


@pytest.fixture
def inject_logger_name(log: pytest_structlog.StructuredLogCapture) -> Generator[None, None, None]:
    # SOURCE: https://github.com/wimglenn/pytest-structlog/blob/1bbad6194989374967c53ff720e36931b15d3081/tests/test_issue22.py#L12
    original_processors = structlog.get_config().get("processors", [])
    if structlog.stdlib.add_logger_name not in original_processors:
        processors = [structlog.stdlib.add_logger_name] + original_processors
        log.original_configure(processors=processors)
        yield
        log.original_configure(processors=original_processors)


@pytest.fixture
def stdlib_configure():
    # SOURCE: https://github.com/wimglenn/pytest-structlog/blob/1bbad6194989374967c53ff720e36931b15d3081/tests/test_issue18.py#L15
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
    )


def _add_process_info(
    logger_name: str | None,
    method_name: str | None,
    event_dict: EventDict,
) -> EventDict:
    """Add process information to the event dictionary.

    Args:
        logger_name: The name of the logger (may be None)
        method_name: The logging method name (may be None)
        event_dict: The event dictionary to process

    Returns:
        EventDict: Modified event dictionary with process information
    """
    event_dict["process_name"] = multiprocessing.current_process().name
    event_dict["process_id"] = multiprocessing.current_process().pid
    return event_dict


def _add_module_name(
    logger_name: str | None | Any,
    method_name: str | None,
    event_dict: EventDict,
) -> EventDict:
    """Add module name to the event dictionary.

    Args:
        logger_name: The name of the logger (may be None)
        method_name: The logging method name (may be None)
        event_dict: The event dictionary to process

    Returns:
        EventDict: Modified event dictionary with module information
    """
    if logger_name is not None:
        if hasattr(logger_name, "name"):
            event_dict["module"] = logger_name.name.split(".")[-1]
        elif isinstance(logger_name, str):
            event_dict["module"] = logger_name.split(".")[-1]
    return event_dict


@pytest.fixture(name="log")
def fixture_log() -> LogCapture:
    """Create a LogCapture fixture for testing.

    Returns:
        LogCapture: A new LogCapture instance
    """
    return LogCapture()


# @pytest.fixture(autouse=True, scope="function")
# def fixture_configure_structlog(log: pytest_structlog.StructuredLogCapture) -> Generator[None, None, None]:
#     """Configure structlog for testing with the LogCapture processor.

#     Args:
#         log: The LogCapture fixture

#     Yields:
#         None
#     """
#     # Reset structlog to defaults before configuring
#     structlog.reset_defaults()

#     # Configure structlog for testing
#     structlog.configure(
#         processors=[
#             # Add stdlib processors first
#             structlog.stdlib.filter_by_level,
#             structlog.stdlib.add_logger_name,
#             structlog.stdlib.add_log_level,
#             structlog.stdlib.PositionalArgumentsFormatter(),
#             # Add structlog processors
#             structlog.processors.TimeStamper(fmt="iso", utc=True),
#             structlog.processors.StackInfoRenderer(),
#             structlog.processors.format_exc_info,
#             # LogCapture must be last to capture processed events
#             log,
#         ],
#         wrapper_class=structlog.stdlib.BoundLogger,
#         context_class=dict,
#         logger_factory=structlog.stdlib.LoggerFactory(),
#         cache_logger_on_first_use=False,  # Important: disable caching for tests
#     )
#     yield
#     structlog.reset_defaults()


# @pytest.fixture(autouse=True, scope="function")
# def clean_logging() -> Generator[None, None, None]:
#     """Reset logging configuration before and after each test.

#     This fixture ensures a clean logging state for each test by:
#     1. Resetting root handlers
#     2. Clearing structlog defaults
#     3. Resetting context variables

#     Yields:
#         None
#     """
#     # Reset before test
#     logging.root.handlers = []
#     structlog.reset_defaults()
#     structlog.contextvars.clear_contextvars()

#     yield

#     # Reset after test
#     logging.root.handlers = []
#     structlog.reset_defaults()
#     structlog.contextvars.clear_contextvars()


def log_exception():
    logger_exc = structlog.get_logger("test_exception_level")
    try:
        1 / 0
    except ZeroDivisionError:
        logger_exc.exception("event_name")


def test_exception_level(stdlib_bound_logger_configure, log: pytest_structlog.StructuredLogCapture) -> None:
    """Test that exception logging works correctly.

    Args:
        stdlib_bound_logger_configure: Fixture to configure structlog
        log: The LogCapture fixture
    """
    log_exception()

    # Verify the log entry
    assert len(log.entries) == 1, "Expected exactly one log entry"
    entry = log.entries[0]
    assert entry["event"] == "event_name"
    assert "exception" in entry, "Missing exception information"
    assert "ZeroDivisionError: division by zero" in entry["exception"]
    assert entry["level"] == "error"
    assert entry["log_level"] == "error"
    assert entry["logger"] == "test_exception_level"


def log_test_message():
    logger = structlog.get_logger("test_logger_initialization")
    logger.info("test message")


# @pytest.mark.skip_until(
#     deadline=datetime.datetime(2025, 1, 25),
#     strict=True,
#     msg="This test can be better",
# )
@pytest.mark.logsonly
@freeze_time("2024-01-01 12:00:00Z")
def test_logger_initialization(log: pytest_structlog.StructuredLogCapture) -> None:
    """Test that the logger is properly initialized.

    This test verifies:
    1. Logger configuration is successful
    2. Logger instance is created correctly
    3. Log messages are captured with correct format
    4. Required logging methods are available
    5. Timestamp format is correct
    6. Log level is properly set

    Args:
        log: The LogCapture fixture for verifying log output
    """
    logger = structlog.get_logger("test_logger_initialization")
    # Configure logging first with test-specific processors
    # structlog.configure(
    #     processors=[
    #         TimeStamper(fmt="iso", utc=True),
    #         add_log_level,
    #         log,
    #     ],
    #     wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    #     cache_logger_on_first_use=False,
    # )

    # Get a logger instance
    # logger = get_logger("test")

    # Verify logger is not None and properly typed
    assert logger is not None
    assert isinstance(logger, (BoundLogger, BindableLogger)), "Logger is not properly configured"

    # Log a test message
    test_message = "test message"
    # logger.info(test_message)

    log_test_message()

    # Get the captured logs
    assert len(log.entries) == 1, "Expected exactly one log message"
    log_entry = log.entries[0]

    # Verify message content
    assert log_entry["event"] == test_message, "Incorrect message content"

    # Verify log level
    assert log_entry["log_level"] == "info", "Incorrect log level"

    # Verify timestamp format (ISO format)
    assert "timestamp" in log_entry, "Missing timestamp"
    assert log_entry["timestamp"] == "2024-01-01T12:00:00Z", "Incorrect timestamp format"

    # Verify logger capabilities
    assert hasattr(logger, "info"), "Logger missing info method"
    assert hasattr(logger, "debug"), "Logger missing debug method"
    assert hasattr(logger, "warning"), "Logger missing warning method"
    assert hasattr(logger, "error"), "Logger missing error method"
    assert all(hasattr(logger, attr) for attr in ["info", "debug", "warning", "error"])


@pytest.mark.logsonly
@pytest.mark.parametrize("enable_json_logs", [True, False])
def test_configure_logging(enable_json_logs: bool, log: pytest_structlog.StructuredLogCapture) -> None:
    """Test the configure_logging function with different output formats.

    Args:
        enable_json_logs: Whether to enable JSON logging
    """
    configure_logging(enable_json_logs=enable_json_logs)
    # logger = get_logger("test")

    # with structlog.testing.capture_logs() as captured:
    #     test_message = "test message"
    #     logger.info(test_message)

    #     assert len(captured) == 1, "Expected exactly one log message"
    #     log_entry = captured[0]
    #     assert log_entry["event"] == test_message, "Incorrect message content"
    #     assert log_entry["log_level"] == "info", "Incorrect log level"

    test_message = "test message"
    logger.info(test_message)

    assert len(log.entries) == 1, "Expected exactly one log message"
    log_entry = log.entries[0]
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
def test_module_name_processor(log: pytest_structlog.StructuredLogCapture) -> None:
    """Test that the module name processor adds correct module information.

    This test verifies both:
    1. Direct processor functionality
    2. Integration with the logging system

    Args:
        log: The LogCapture fixture for verifying log output
    """
    # Test the processor function directly
    test_event = {"event": "test message"}
    processed_event = _add_module_name("test_module", "info", test_event)

    assert "module" in processed_event, "Processor failed to add module field"
    assert processed_event["module"] == "test_module", "Processor added incorrect module name"
    assert processed_event["event"] == "test message", "Processor modified event message"

    # # Test the processor integrated in the logging system
    # structlog.configure(
    #     processors=[
    #         TimeStamper(fmt="iso", utc=True),
    #         add_log_level,
    #         structlog.stdlib.add_logger_name,
    #         _add_module_name,
    #         log,
    #     ],
    #     wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    #     logger_factory=structlog.stdlib.LoggerFactory(),
    #     cache_logger_on_first_use=False,
    # )

    logger = get_logger("test_module")
    test_message = "integration test message"
    logger.info(test_message)

    assert len(log.entries) == 1, "Expected exactly one log message"
    log_entry = log.entries[0]

    # Verify logger name is preserved
    assert "logger" in log_entry, "Missing logger name"
    assert log_entry["logger"] == "test_module", "Missing or incorrect logger name"

    # Verify module name is added
    assert "module" in log_entry, "Missing module name"
    assert log_entry["module"] == "test_module", "Missing or incorrect module name"

    # Verify original message is preserved
    assert log_entry["event"] == test_message, "Log message was modified"


@pytest.mark.logsonly
@freeze_time("2024-01-01 12:00:00Z")
def test_timestamp_formatting(log: pytest_structlog.StructuredLogCapture) -> None:
    """Test that timestamps are properly formatted.

    This test verifies:
    1. Timestamps are in ISO format
    2. Timestamps are in UTC
    3. Timestamps are added to all log entries

    Args:
        log: The LogCapture fixture for verifying log output
    """
    # # Reset and reconfigure for this test
    # structlog.reset_defaults()
    # structlog.configure(
    #     processors=[
    #         structlog.stdlib.filter_by_level,
    #         structlog.stdlib.add_logger_name,
    #         structlog.stdlib.add_log_level,
    #         structlog.processors.TimeStamper(fmt="iso", utc=True),
    #         structlog.processors.StackInfoRenderer(),
    #         log,
    #     ],
    #     wrapper_class=structlog.stdlib.BoundLogger,
    #     context_class=dict,
    #     logger_factory=structlog.stdlib.LoggerFactory(),
    #     cache_logger_on_first_use=False,
    # )

    logger = get_logger("test_timestamps")

    logger.info("test message")
    assert len(log.entries) == 1, "Expected exactly one log message"
    log_entry = log.entries[0]

    # Verify timestamp exists and format
    assert "timestamp" in log_entry, "Missing timestamp"
    assert log_entry["timestamp"] == "2024-01-01T12:00:00Z", "Incorrect timestamp format"


@pytest.mark.logsonly
def test_error_formatting(log: pytest_structlog.StructuredLogCapture) -> None:
    """Test that errors are properly formatted in log entries.

    This test verifies:
    1. Exception information is captured and formatted
    2. Stack traces are included
    3. Error context is preserved

    Args:
        log: The LogCapture fixture for verifying log output
    """
    # # Reset and reconfigure for this test
    # structlog.reset_defaults()
    # structlog.configure(
    #     processors=[
    #         # Add stdlib processors first
    #         structlog.stdlib.filter_by_level,
    #         structlog.stdlib.add_logger_name,
    #         structlog.stdlib.add_log_level,
    #         structlog.stdlib.PositionalArgumentsFormatter(),
    #         # Add structlog processors
    #         structlog.processors.TimeStamper(fmt="iso", utc=True),
    #         structlog.processors.StackInfoRenderer(),
    #         structlog.processors.format_exc_info,
    #         # LogCapture must be last
    #         log,
    #     ],
    #     wrapper_class=structlog.stdlib.BoundLogger,
    #     context_class=dict,
    #     logger_factory=structlog.stdlib.LoggerFactory(),
    #     cache_logger_on_first_use=False,
    # )

    logger = get_logger("test_errors")

    try:
        raise ValueError("test error")
    except ValueError:
        logger.exception("error occurred")

    assert len(log.entries) == 1, "Expected exactly one log message"
    log_entry = log.entries[0]

    # Verify error details
    assert "exception" in log_entry, "Missing exception info"
    assert isinstance(log_entry["exception"], str), "Exception should be formatted as string"
    assert "Traceback (most recent call last)" in log_entry["exception"], "Missing traceback"
    assert "ValueError: test error" in log_entry["exception"], "Missing error message"
    assert log_entry["log_level"] == "error", "Incorrect log level"


@pytest.mark.logsonly
def test_context_preservation(log: pytest_structlog.StructuredLogCapture) -> None:
    """Test that context is properly preserved across log entries.

    This test verifies:
    1. Bound context is preserved
    2. Context is properly merged
    3. Context doesn't leak between loggers

    Args:
        log: The LogCapture fixture for verifying log output
    """
    configure_logging(enable_json_logs=True)
    logger = get_logger("test_context")

    # Test with bound context
    bound_logger = logger.bind(user_id="123", session="abc")

    with structlog.testing.capture_logs() as captured:
        bound_logger.info("test message")

        assert len(captured) == 1, "Expected exactly one log message"
        log_entry = captured[0]

        # Verify context is preserved
        assert log_entry["user_id"] == "123", "Missing or incorrect user_id context"
        assert log_entry["session"] == "abc", "Missing or incorrect session context"

        # Verify original logger doesn't have the context
        logger.info("another message")
        assert "user_id" not in captured[1], "Context leaked to original logger"


@pytest.mark.logsonly
def test_processor_chain_ordering(log: pytest_structlog.StructuredLogCapture) -> None:
    """Test that the processor chain executes in the correct order.

    This test verifies:
    1. Processors are executed in the specified order
    2. Each processor properly modifies the event dict
    3. Final output contains all expected fields

    Args:
        log: The LogCapture fixture for verifying log output
    """

    # # Create processors that track execution order
    # def first_processor(
    #     logger_name: str | None,
    #     method_name: str | None,
    #     event_dict: EventDict,
    # ) -> EventDict:
    #     """Add first processor tracking."""
    #     event_dict.setdefault("processor_order", []).append("first")
    #     return event_dict

    # def second_processor(
    #     logger_name: str | None,
    #     method_name: str | None,
    #     event_dict: EventDict,
    # ) -> EventDict:
    #     """Add second processor tracking."""
    #     event_dict.setdefault("processor_order", []).append("second")
    #     return event_dict

    # # Reset and reconfigure for this test
    # structlog.reset_defaults()
    # structlog.configure(
    #     processors=[
    #         # Our tracking processors
    #         first_processor,
    #         second_processor,
    #         # Standard processors
    #         structlog.stdlib.filter_by_level,
    #         structlog.stdlib.add_logger_name,
    #         structlog.stdlib.add_log_level,
    #         structlog.stdlib.PositionalArgumentsFormatter(),
    #         structlog.processors.TimeStamper(fmt="iso", utc=True),
    #         # LogCapture must be last
    #         log,
    #     ],
    #     wrapper_class=structlog.stdlib.BoundLogger,
    #     context_class=dict,
    #     logger_factory=structlog.stdlib.LoggerFactory(),
    #     cache_logger_on_first_use=False,
    # )

    logger = get_logger("test_processors")
    logger.info("test message")

    assert len(log.entries) == 1, "Expected exactly one log message"
    log_entry = log.entries[0]

    # Verify processor execution order
    assert "processor_order" in log_entry, "Missing processor order tracking"
    assert log_entry["processor_order"] == ["first", "second"], "Incorrect processor order"

    # Verify standard processors executed
    assert "timestamp" in log_entry, "TimeStamper processor didn't execute"
    assert "log_level" in log_entry, "add_log_level processor didn't execute"
    assert "logger" in log_entry, "add_logger_name processor didn't execute"
