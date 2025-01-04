"""Tests for GetCurrentTimeTool."""

from __future__ import annotations

import re

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from pytest_mock import MockerFixture

from democracy_exe.agentic.tools.get_current_time_tool import GetCurrentTimeResponse, GetCurrentTimeTool


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def get_current_time_tool() -> GetCurrentTimeTool:
    """Create GetCurrentTimeTool instance for testing.

    Returns:
        GetCurrentTimeTool instance
    """
    return GetCurrentTimeTool()


@pytest.fixture
def mock_datetime(mocker: MockerFixture) -> datetime:
    """Mock datetime.now() for consistent testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mocked datetime object
    """
    mock_time = datetime(2024, 1, 1, 12, 0, 0)
    mocker.patch("democracy_exe.agentic.tools.get_current_time_tool.datetime", autospec=True)
    mocker.patch("democracy_exe.agentic.tools.get_current_time_tool.datetime.now", return_value=mock_time)
    return mock_time


def test_get_time_default_format(
    get_current_time_tool: GetCurrentTimeTool, mock_datetime: datetime, caplog: LogCaptureFixture
) -> None:
    """Test getting time with default format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        mock_datetime: Mocked datetime object
        caplog: Pytest fixture for capturing log messages
    """
    formatted_time, timestamp = get_current_time_tool._get_time()

    # Verify formatted time
    assert formatted_time == "2024-01-01 12:00:00"
    assert timestamp == mock_datetime.timestamp()

    # Verify logging
    assert "Getting current time with format" in caplog.text
    assert str(formatted_time) in caplog.text


def test_get_time_custom_format(
    get_current_time_tool: GetCurrentTimeTool, mock_datetime: datetime, caplog: LogCaptureFixture
) -> None:
    """Test getting time with custom format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        mock_datetime: Mocked datetime object
        caplog: Pytest fixture for capturing log messages
    """
    formatted_time, timestamp = get_current_time_tool._get_time("%H:%M:%S")

    # Verify formatted time
    assert formatted_time == "12:00:00"
    assert timestamp == mock_datetime.timestamp()

    # Verify logging
    assert "Getting current time with format" in caplog.text
    assert str(formatted_time) in caplog.text


def test_get_time_invalid_format(
    get_current_time_tool: GetCurrentTimeTool, mock_datetime: datetime, caplog: LogCaptureFixture
) -> None:
    """Test getting time with invalid format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        mock_datetime: Mocked datetime object
        caplog: Pytest fixture for capturing log messages
    """
    with pytest.raises(ValueError, match="Invalid time format"):
        get_current_time_tool._get_time("invalid")

    # Verify logging
    assert "Error formatting time" in caplog.text


def test_run_success(
    get_current_time_tool: GetCurrentTimeTool, mock_datetime: datetime, caplog: LogCaptureFixture
) -> None:
    """Test successful synchronous time retrieval.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        mock_datetime: Mocked datetime object
        caplog: Pytest fixture for capturing log messages
    """
    result = get_current_time_tool.run({})

    # Verify response
    assert result["current_time"] == "2024-01-01 12:00:00"
    assert result["timestamp"] == mock_datetime.timestamp()
    assert result.get("error") is None

    # Verify logging
    assert "Getting current time synchronously" in caplog.text
    assert "Successfully retrieved current time" in caplog.text


def test_run_custom_format(
    get_current_time_tool: GetCurrentTimeTool, mock_datetime: datetime, caplog: LogCaptureFixture
) -> None:
    """Test time retrieval with custom format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        mock_datetime: Mocked datetime object
        caplog: Pytest fixture for capturing log messages
    """
    result = get_current_time_tool.run({"format": "%Y-%m-%d"})

    # Verify response
    assert result["current_time"] == "2024-01-01"
    assert result["timestamp"] == mock_datetime.timestamp()
    assert result.get("error") is None

    # Verify logging
    assert "Getting current time synchronously" in caplog.text
    assert "Successfully retrieved current time" in caplog.text


def test_run_invalid_format(
    get_current_time_tool: GetCurrentTimeTool, mock_datetime: datetime, caplog: LogCaptureFixture
) -> None:
    """Test time retrieval with invalid format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        mock_datetime: Mocked datetime object
        caplog: Pytest fixture for capturing log messages
    """
    result = get_current_time_tool.run({"format": "invalid"})

    # Verify error response
    assert result["current_time"] == ""
    assert result["timestamp"] == 0.0
    assert "Invalid time format" in result["error"]

    # Verify logging
    assert "Failed to get current time" in caplog.text


@pytest.mark.asyncio
async def test_arun_success(
    get_current_time_tool: GetCurrentTimeTool, mock_datetime: datetime, caplog: LogCaptureFixture
) -> None:
    """Test successful asynchronous time retrieval.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        mock_datetime: Mocked datetime object
        caplog: Pytest fixture for capturing log messages
    """
    result = await get_current_time_tool.arun({})

    # Verify response
    assert result["current_time"] == "2024-01-01 12:00:00"
    assert result["timestamp"] == mock_datetime.timestamp()
    assert result.get("error") is None

    # Verify logging
    assert "Getting current time asynchronously" in caplog.text
    assert "Successfully retrieved current time" in caplog.text


@pytest.mark.asyncio
async def test_arun_custom_format(
    get_current_time_tool: GetCurrentTimeTool, mock_datetime: datetime, caplog: LogCaptureFixture
) -> None:
    """Test async time retrieval with custom format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        mock_datetime: Mocked datetime object
        caplog: Pytest fixture for capturing log messages
    """
    result = await get_current_time_tool.arun({"format": "%H:%M:%S"})

    # Verify response
    assert result["current_time"] == "12:00:00"
    assert result["timestamp"] == mock_datetime.timestamp()
    assert result.get("error") is None

    # Verify logging
    assert "Getting current time asynchronously" in caplog.text
    assert "Successfully retrieved current time" in caplog.text


@pytest.mark.asyncio
async def test_arun_invalid_format(
    get_current_time_tool: GetCurrentTimeTool, mock_datetime: datetime, caplog: LogCaptureFixture
) -> None:
    """Test async time retrieval with invalid format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        mock_datetime: Mocked datetime object
        caplog: Pytest fixture for capturing log messages
    """
    result = await get_current_time_tool.arun({"format": "invalid"})

    # Verify error response
    assert result["current_time"] == ""
    assert result["timestamp"] == 0.0
    assert "Invalid time format" in result["error"]

    # Verify logging
    assert "Failed to get current time" in caplog.text
