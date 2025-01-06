from __future__ import annotations

import logging

from datetime import datetime

from freezegun import freeze_time
from loguru import logger

import pytest

from democracy_exe.bot_logger import InterceptHandlerImproved


@pytest.fixture
def intercept_handler():
    """Fixture to create an InterceptHandlerImproved instance."""
    return InterceptHandlerImproved()


@pytest.fixture
def log_record():
    """Fixture to create a basic LogRecord."""
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )
    return record


@pytest.mark.unit
def test_emit_basic_message(intercept_handler, log_record, caplog):
    """Test basic message emission through InterceptHandlerImproved."""
    with caplog.at_level(logging.INFO):
        intercept_handler.emit(log_record)
        assert "Test message" in caplog.text


@pytest.mark.unit
def test_emit_with_exception(intercept_handler, caplog):
    """Test message emission with exception information."""
    try:
        raise ValueError("Test exception")
    except ValueError:
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test_file.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=True,
        )
        with caplog.at_level(logging.ERROR):
            intercept_handler.emit(record)
            assert "Error occurred" in caplog.text
            assert "ValueError: Test exception" in caplog.text


@pytest.mark.unit
def test_emit_different_log_levels(intercept_handler, caplog):
    """Test emission of messages at different log levels."""
    levels = [
        (logging.DEBUG, "DEBUG message"),
        (logging.INFO, "INFO message"),
        (logging.WARNING, "WARNING message"),
        (logging.ERROR, "ERROR message"),
        (logging.CRITICAL, "CRITICAL message"),
    ]

    for level, msg in levels:
        record = logging.LogRecord(
            name="test_logger", level=level, pathname="test_file.py", lineno=42, msg=msg, args=(), exc_info=None
        )
        with caplog.at_level(level):
            intercept_handler.emit(record)
            assert msg in caplog.text


@pytest.mark.unit
def test_emit_with_formatted_message(intercept_handler, caplog):
    """Test emission of formatted messages."""
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test_file.py",
        lineno=42,
        msg="Test %s with %d",
        args=("message", 42),
        exc_info=None,
    )
    with caplog.at_level(logging.INFO):
        intercept_handler.emit(record)
        assert "Test message with 42" in caplog.text


@pytest.mark.unit
def test_emit_with_custom_level(intercept_handler, caplog):
    """Test emission with a custom log level."""
    CUSTOM_LEVEL = 15
    logging.addLevelName(CUSTOM_LEVEL, "CUSTOM")

    record = logging.LogRecord(
        name="test_logger",
        level=CUSTOM_LEVEL,
        pathname="test_file.py",
        lineno=42,
        msg="Custom level message",
        args=(),
        exc_info=None,
    )
    with caplog.at_level(CUSTOM_LEVEL):
        intercept_handler.emit(record)
        assert "Custom level message" in caplog.text


@pytest.mark.unit
@freeze_time("2024-01-06 12:34:56.789")
def test_format_record():
    """Test that format_record includes all required components."""
    from democracy_exe.bot_logger import format_record

    # Create a test record with all components
    record = {
        "time": datetime.now(),
        "level": {"name": "INFO", "no": 20, "icon": "🔵"},
        "name": "test_logger",
        "function": "test_func",
        "file": "test_file.py",
        "line": 42,
        "message": "Test message",
        "exception": None,
        "extra": {"payload": {"key": "value"}, "request_id": "test-123"},
    }

    # Format the record using format_record
    formatted = format_record(record)

    # Expected components in specific order
    expected_components = [
        "2024-01-06 12:34:56.789",  # time
        "INFO",  # level
        "test_logger",  # name
        "test_func",  # function
        "test_file.py:42",  # file and line
        "Test message",  # message
        "'key': 'value'",  # payload in extra
    ]

    # Verify all components are present in order
    formatted_lower = formatted.lower()
    last_pos = 0
    for component in expected_components:
        pos = formatted_lower.find(component.lower())
        assert pos != -1, f"Component not found: {component}"
        assert pos >= last_pos, f"Component out of order: {component}"
        last_pos = pos
