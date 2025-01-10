"""Tests for GetRandomNumberTool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from structlog.testing import capture_logs

import pytest

from pytest_mock import MockerFixture

from democracy_exe.agentic.tools.get_random_number_tool import GetRandomNumberResponse, GetRandomNumberTool


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

logger = structlog.get_logger(__name__)


@pytest.fixture
def get_random_number_tool() -> GetRandomNumberTool:
    """Create GetRandomNumberTool instance for testing.

    Returns:
        GetRandomNumberTool instance
    """
    return GetRandomNumberTool()


@pytest.mark.toolonly
def test_generate_number_with_seed(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test generating random number with seed.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        min_val = 1
        max_val = 100
        seed = 42
        result = get_random_number_tool._generate_number(min_val, max_val, seed)

        # Verify result is within range
        assert min_val <= result <= max_val

        # Debug: Print captured logs
        print("\nCaptured logs in test_generate_number_with_seed:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Generating random number with seed" for log in captured), (
            "Expected 'Generating random number with seed' message not found in logs"
        )

        assert any(log.get("event") == f"Generated number: {result}" for log in captured), (
            "Expected 'Generated number' message not found in logs"
        )


@pytest.mark.toolonly
def test_generate_number_without_seed(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test generating random number without seed.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        min_val = 1
        max_val = 100
        result = get_random_number_tool._generate_number(min_val, max_val)

        # Verify result is within range
        assert min_val <= result <= max_val

        # Debug: Print captured logs
        print("\nCaptured logs in test_generate_number_without_seed:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Generating random number without seed" for log in captured), (
            "Expected 'Generating random number without seed' message not found in logs"
        )

        assert any(log.get("event") == f"Generated number: {result}" for log in captured), (
            "Expected 'Generated number' message not found in logs"
        )


@pytest.mark.toolonly
def test_generate_number_invalid_range(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test generating random number with invalid range.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        min_val = 100
        max_val = 1  # max < min

        with pytest.raises(ValueError, match="Maximum value must be greater than minimum value"):
            get_random_number_tool._generate_number(min_val, max_val)

        # Debug: Print captured logs
        print("\nCaptured logs in test_generate_number_invalid_range:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Invalid range for random number generation" for log in captured), (
            "Expected 'Invalid range for random number generation' message not found in logs"
        )


@pytest.mark.toolonly
def test_run_success(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test successful synchronous number generation.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        result = get_random_number_tool.run({"min_value": 1, "max_value": 100})

        # Verify response
        assert isinstance(result["random_number"], int)
        assert 1 <= result["random_number"] <= 100
        assert result["min_value"] == 1
        assert result["max_value"] == 100
        assert result.get("error") is None
        assert result.get("seed") is None

        # Debug: Print captured logs
        print("\nCaptured logs in test_run_success:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Generating random number synchronously" for log in captured), (
            "Expected 'Generating random number synchronously' message not found in logs"
        )

        assert any(
            log.get("event") == f"Successfully generated random number: {result['random_number']}" for log in captured
        ), "Expected 'Successfully generated random number' message not found in logs"


@pytest.mark.toolonly
def test_run_with_seed(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test number generation with seed.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        seed = 42
        result1 = get_random_number_tool.run({"min_value": 1, "max_value": 100, "seed": seed})
        result2 = get_random_number_tool.run({"min_value": 1, "max_value": 100, "seed": seed})

        # Verify responses are identical with same seed
        assert result1["random_number"] == result2["random_number"]
        assert result1["seed"] == result2["seed"] == seed

        # Debug: Print captured logs
        print("\nCaptured logs in test_run_with_seed:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Generating random number synchronously" for log in captured), (
            "Expected 'Generating random number synchronously' message not found in logs"
        )

        assert any(
            log.get("event") == f"Successfully generated random number: {result1['random_number']}" for log in captured
        ), "Expected 'Successfully generated random number' message not found in logs"


@pytest.mark.toolonly
def test_run_invalid_range(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test number generation with invalid range.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        result = get_random_number_tool.run({
            "min_value": 100,
            "max_value": 1,  # max < min
        })

        # Verify error response
        assert result["random_number"] == 0
        assert result["min_value"] == 100
        assert result["max_value"] == 1
        assert "Maximum value must be greater than minimum value" in result["error"]

        # Debug: Print captured logs
        print("\nCaptured logs in test_run_invalid_range:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Generating random number synchronously" for log in captured), (
            "Expected 'Generating random number synchronously' message not found in logs"
        )

        assert any(log.get("event") == "Invalid range for random number generation" for log in captured), (
            "Expected 'Invalid range for random number generation' message not found in logs"
        )


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_success(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test successful asynchronous number generation.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        result = await get_random_number_tool.arun({"min_value": 1, "max_value": 100})

        # Verify response
        assert isinstance(result["random_number"], int)
        assert 1 <= result["random_number"] <= 100
        assert result["min_value"] == 1
        assert result["max_value"] == 100
        assert result.get("error") is None
        assert result.get("seed") is None

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_success:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Generating random number asynchronously" for log in captured), (
            "Expected 'Generating random number asynchronously' message not found in logs"
        )

        assert any(
            log.get("event") == f"Successfully generated random number: {result['random_number']}" for log in captured
        ), "Expected 'Successfully generated random number' message not found in logs"


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_with_seed(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test asynchronous number generation with seed.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        seed = 42
        result1 = await get_random_number_tool.arun({"min_value": 1, "max_value": 100, "seed": seed})
        result2 = await get_random_number_tool.arun({"min_value": 1, "max_value": 100, "seed": seed})

        # Verify responses are identical with same seed
        assert result1["random_number"] == result2["random_number"]
        assert result1["seed"] == result2["seed"] == seed

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_with_seed:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Generating random number asynchronously" for log in captured), (
            "Expected 'Generating random number asynchronously' message not found in logs"
        )

        assert any(
            log.get("event") == f"Successfully generated random number: {result1['random_number']}" for log in captured
        ), "Expected 'Successfully generated random number' message not found in logs"


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_invalid_range(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test asynchronous number generation with invalid range.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        result = await get_random_number_tool.arun({
            "min_value": 100,
            "max_value": 1,  # max < min
        })

        # Verify error response
        assert result["random_number"] == 0
        assert result["min_value"] == 100
        assert result["max_value"] == 1
        assert "Maximum value must be greater than minimum value" in result["error"]

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_invalid_range:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Generating random number asynchronously" for log in captured), (
            "Expected 'Generating random number asynchronously' message not found in logs"
        )

        assert any(log.get("event") == "Invalid range for random number generation" for log in captured), (
            "Expected 'Invalid range for random number generation' message not found in logs"
        )
