"""Tests for GetCurrentTimeTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from structlog.testing import capture_logs

import pytest

from pytest_mock import MockerFixture

from democracy_exe.agentic.tools.get_current_time_tool import GetCurrentTimeResponse, GetCurrentTimeTool


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

logger = structlog.get_logger(__name__)


@pytest.fixture
def get_current_time_tool() -> GetCurrentTimeTool:
    """Create GetCurrentTimeTool instance for testing.

    Returns:
        GetCurrentTimeTool instance
    """
    return GetCurrentTimeTool()


@pytest.mark.toolonly
def test_get_time_default_format(get_current_time_tool: GetCurrentTimeTool, caplog: LogCaptureFixture) -> None:
    """Test getting current time with default format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        formatted_time, timestamp = get_current_time_tool._get_time()

        # Verify result format
        assert isinstance(formatted_time, str)
        assert isinstance(timestamp, float)

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_time_default_format:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Getting current time with format: %Y-%m-%d %H:%M:%S" for log in captured), (
            "Expected 'Getting current time with format' message not found in logs"
        )

        assert any(log.get("event").startswith("Current time:") for log in captured), (
            "Expected 'Current time' message not found in logs"
        )


@pytest.mark.toolonly
def test_get_time_custom_format(get_current_time_tool: GetCurrentTimeTool, caplog: LogCaptureFixture) -> None:
    """Test getting current time with custom format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        custom_format = "%Y-%m-%d"
        formatted_time, timestamp = get_current_time_tool._get_time(custom_format)

        # Verify result format
        assert isinstance(formatted_time, str)
        assert len(formatted_time) == 10  # YYYY-MM-DD format
        assert isinstance(timestamp, float)

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_time_custom_format:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == f"Getting current time with format: {custom_format}" for log in captured), (
            "Expected 'Getting current time with format' message not found in logs"
        )

        assert any(log.get("event").startswith("Current time:") for log in captured), (
            "Expected 'Current time' message not found in logs"
        )


@pytest.mark.toolonly
def test_get_time_invalid_format(get_current_time_tool: GetCurrentTimeTool, caplog: LogCaptureFixture) -> None:
    """Test getting current time with invalid format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        invalid_format = "invalid"
        formatted_time, timestamp = get_current_time_tool._get_time(invalid_format)

        # Verify result format
        assert formatted_time == invalid_format
        assert isinstance(timestamp, float)

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_time_invalid_format:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == f"Getting current time with format: {invalid_format}" for log in captured), (
            "Expected 'Getting current time with format' message not found in logs"
        )

        assert any(log.get("event").startswith("Current time:") for log in captured), (
            "Expected 'Current time' message not found in logs"
        )


@pytest.mark.toolonly
def test_run_success(get_current_time_tool: GetCurrentTimeTool, caplog: LogCaptureFixture) -> None:
    """Test successful synchronous time retrieval.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        result = get_current_time_tool.run({})

        # Verify response
        assert isinstance(result["current_time"], str)
        assert isinstance(result["timestamp"], float)
        assert result.get("error") is None

        # Debug: Print captured logs
        print("\nCaptured logs in test_run_success:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Getting current time synchronously" for log in captured), (
            "Expected 'Getting current time synchronously' message not found in logs"
        )

        assert any(log.get("event").startswith("Successfully retrieved current time:") for log in captured), (
            "Expected 'Successfully retrieved current time' message not found in logs"
        )


@pytest.mark.toolonly
def test_run_custom_format(get_current_time_tool: GetCurrentTimeTool, caplog: LogCaptureFixture) -> None:
    """Test time retrieval with custom format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        custom_format = "%Y-%m-%d"
        result = get_current_time_tool.run({"format": custom_format})

        # Verify response
        assert isinstance(result["current_time"], str)
        assert len(result["current_time"]) == 10  # YYYY-MM-DD format
        assert isinstance(result["timestamp"], float)
        assert result.get("error") is None

        # Debug: Print captured logs
        print("\nCaptured logs in test_run_custom_format:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Getting current time synchronously" for log in captured), (
            "Expected 'Getting current time synchronously' message not found in logs"
        )

        assert any(log.get("event").startswith("Successfully retrieved current time:") for log in captured), (
            "Expected 'Successfully retrieved current time' message not found in logs"
        )


@pytest.mark.toolonly
def test_run_invalid_format(get_current_time_tool: GetCurrentTimeTool, caplog: LogCaptureFixture) -> None:
    """Test time retrieval with invalid format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        invalid_format = "invalid"
        result = get_current_time_tool.run({"format": invalid_format})

        # Verify error response
        assert result["current_time"] == invalid_format
        assert isinstance(result["timestamp"], float)
        assert not result["error"]

        # Debug: Print captured logs
        print("\nCaptured logs in test_run_invalid_format:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Getting current time synchronously" for log in captured), (
            "Expected 'Getting current time synchronously' message not found in logs"
        )

        assert any(log.get("event").startswith("Successfully retrieved current time:") for log in captured), (
            "Expected 'Successfully retrieved current time' message not found in logs"
        )


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_success(get_current_time_tool: GetCurrentTimeTool, caplog: LogCaptureFixture) -> None:
    """Test successful asynchronous time retrieval.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        result = await get_current_time_tool.arun({})

        # Verify response
        assert isinstance(result["current_time"], str)
        assert isinstance(result["timestamp"], float)
        assert result.get("error") is None

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_success:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Getting current time asynchronously" for log in captured), (
            "Expected 'Getting current time asynchronously' message not found in logs"
        )

        assert any(log.get("event").startswith("Successfully retrieved current time:") for log in captured), (
            "Expected 'Successfully retrieved current time' message not found in logs"
        )


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_custom_format(get_current_time_tool: GetCurrentTimeTool, caplog: LogCaptureFixture) -> None:
    """Test async time retrieval with custom format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        custom_format = "%H:%M:%S"
        result = await get_current_time_tool.arun({"format": custom_format})

        # Verify response
        assert isinstance(result["current_time"], str)
        assert len(result["current_time"]) == 8  # HH:MM:SS format
        assert isinstance(result["timestamp"], float)
        assert result.get("error") is None

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_custom_format:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Getting current time asynchronously" for log in captured), (
            "Expected 'Getting current time asynchronously' message not found in logs"
        )

        assert any(log.get("event").startswith("Successfully retrieved current time:") for log in captured), (
            "Expected 'Successfully retrieved current time' message not found in logs"
        )


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_invalid_format(get_current_time_tool: GetCurrentTimeTool, caplog: LogCaptureFixture) -> None:
    """Test async time retrieval with invalid format.

    Args:
        get_current_time_tool: GetCurrentTimeTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        invalid_format = "invalid"
        result = await get_current_time_tool.arun({"format": invalid_format})

        # Verify error response
        assert result["current_time"] == invalid_format
        assert isinstance(result["timestamp"], float)
        assert not result["error"]

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_invalid_format:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Getting current time asynchronously" for log in captured), (
            "Expected 'Getting current time asynchronously' message not found in logs"
        )

        assert any(log.get("event").startswith("Successfully retrieved current time:") for log in captured), (
            "Expected 'Successfully retrieved current time' message not found in logs"
        )
