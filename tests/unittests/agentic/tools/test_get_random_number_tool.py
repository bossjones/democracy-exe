"""Tests for GetRandomNumberTool."""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

import pytest

from pytest_mock import MockerFixture

from democracy_exe.agentic.tools.get_random_number_tool import GetRandomNumberResponse, GetRandomNumberTool


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def get_random_number_tool() -> GetRandomNumberTool:
    """Create GetRandomNumberTool instance for testing.

    Returns:
        GetRandomNumberTool instance
    """
    return GetRandomNumberTool()


@pytest.fixture
def mock_random(mocker: MockerFixture) -> None:
    """Mock random.randint for consistent testing.

    Args:
        mocker: Pytest mocker fixture
    """
    mocker.patch("random.randint", return_value=42)


def test_generate_number_with_seed(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test generating number with seed.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    # Use a specific seed for reproducibility
    seed = 12345
    min_value = 1
    max_value = 100

    # Generate first number
    random_number = get_random_number_tool._generate_number(min_value, max_value, seed)

    # Generate second number with same seed - should be identical
    random_number2 = get_random_number_tool._generate_number(min_value, max_value, seed)

    assert random_number == random_number2
    assert min_value <= random_number <= max_value

    # Verify logging
    assert "Generating random number with seed" in caplog.text
    assert str(random_number) in caplog.text


def test_generate_number_without_seed(
    get_random_number_tool: GetRandomNumberTool, mock_random: None, caplog: LogCaptureFixture
) -> None:
    """Test generating number without seed.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        mock_random: Fixture that mocks random.randint
        caplog: Pytest fixture for capturing log messages
    """
    min_value = 1
    max_value = 100

    random_number = get_random_number_tool._generate_number(min_value, max_value)

    assert random_number == 42  # Value from mock
    assert min_value <= random_number <= max_value

    # Verify logging
    assert "Generating random number without seed" in caplog.text
    assert str(random_number) in caplog.text


def test_generate_number_invalid_range(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test generating number with invalid range.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    min_value = 100
    max_value = 1  # max < min

    with pytest.raises(ValueError, match="Maximum value must be greater than minimum value"):
        get_random_number_tool._generate_number(min_value, max_value)

    # Verify logging
    assert "Invalid range for random number generation" in caplog.text


def test_run_success(get_random_number_tool: GetRandomNumberTool, mock_random: None, caplog: LogCaptureFixture) -> None:
    """Test successful synchronous number generation.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        mock_random: Fixture that mocks random.randint
        caplog: Pytest fixture for capturing log messages
    """
    result = get_random_number_tool.run({"min_value": 1, "max_value": 100})

    # Verify response
    assert result["random_number"] == 42  # Value from mock
    assert result["min_value"] == 1
    assert result["max_value"] == 100
    assert result.get("error") is None
    assert result.get("seed") is None

    # Verify logging
    assert "Generating random number synchronously" in caplog.text
    assert "Successfully generated random number" in caplog.text


def test_run_with_seed(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test number generation with seed.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    seed = 12345
    result1 = get_random_number_tool.run({"min_value": 1, "max_value": 100, "seed": seed})

    result2 = get_random_number_tool.run({"min_value": 1, "max_value": 100, "seed": seed})

    # Verify responses are identical with same seed
    assert result1["random_number"] == result2["random_number"]
    assert result1["seed"] == result2["seed"] == seed

    # Verify logging
    assert "Generating random number synchronously" in caplog.text
    assert "Successfully generated random number" in caplog.text


def test_run_invalid_range(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test number generation with invalid range.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    result = get_random_number_tool.run({
        "min_value": 100,
        "max_value": 1,  # max < min
    })

    # Verify error response
    assert result["random_number"] == 0
    assert result["min_value"] == 100
    assert result["max_value"] == 1
    assert "Maximum value must be greater than minimum value" in result["error"]

    # Verify logging
    assert "Failed to generate random number" in caplog.text


@pytest.mark.asyncio
async def test_arun_success(
    get_random_number_tool: GetRandomNumberTool, mock_random: None, caplog: LogCaptureFixture
) -> None:
    """Test successful asynchronous number generation.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        mock_random: Fixture that mocks random.randint
        caplog: Pytest fixture for capturing log messages
    """
    result = await get_random_number_tool.arun({"min_value": 1, "max_value": 100})

    # Verify response
    assert result["random_number"] == 42  # Value from mock
    assert result["min_value"] == 1
    assert result["max_value"] == 100
    assert result.get("error") is None
    assert result.get("seed") is None

    # Verify logging
    assert "Generating random number asynchronously" in caplog.text
    assert "Successfully generated random number" in caplog.text


@pytest.mark.asyncio
async def test_arun_with_seed(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test async number generation with seed.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    seed = 12345
    result1 = await get_random_number_tool.arun({"min_value": 1, "max_value": 100, "seed": seed})

    result2 = await get_random_number_tool.arun({"min_value": 1, "max_value": 100, "seed": seed})

    # Verify responses are identical with same seed
    assert result1["random_number"] == result2["random_number"]
    assert result1["seed"] == result2["seed"] == seed

    # Verify logging
    assert "Generating random number asynchronously" in caplog.text
    assert "Successfully generated random number" in caplog.text


@pytest.mark.asyncio
async def test_arun_invalid_range(get_random_number_tool: GetRandomNumberTool, caplog: LogCaptureFixture) -> None:
    """Test async number generation with invalid range.

    Args:
        get_random_number_tool: GetRandomNumberTool instance
        caplog: Pytest fixture for capturing log messages
    """
    result = await get_random_number_tool.arun({
        "min_value": 100,
        "max_value": 1,  # max < min
    })

    # Verify error response
    assert result["random_number"] == 0
    assert result["min_value"] == 100
    assert result["max_value"] == 1
    assert "Maximum value must be greater than minimum value" in result["error"]

    # Verify logging
    assert "Failed to generate random number" in caplog.text
