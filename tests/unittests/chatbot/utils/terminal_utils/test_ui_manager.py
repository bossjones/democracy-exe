# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Unit tests for the terminal UI manager."""

from __future__ import annotations

import datetime
import sys

from collections.abc import Generator
from typing import Any

import pytest

from pytest_mock import MockerFixture

from democracy_exe.chatbot.utils.terminal_utils import UIManager


@pytest.fixture
def mock_stderr(mocker: MockerFixture) -> Generator[MockerFixture, None, None]:
    """Mock stderr for testing.

    Args:
        mocker: Pytest mocker fixture

    Yields:
        MockerFixture: Mocked stderr
    """
    # Create a properly configured mock
    mock = mocker.MagicMock(name="stderr")
    mock.flush = mocker.Mock(name="flush")
    mock.write = mocker.Mock(name="write")

    # Patch both module-level and global sys.stderr
    mocker.patch("sys.stderr", mock)
    mocker.patch("democracy_exe.chatbot.utils.terminal_utils.ui_manager.sys.stderr", mock)

    yield mock


@pytest.fixture
def ui_manager(mock_stderr: MockerFixture) -> UIManager:
    """Create a UI manager instance for testing.

    Args:
        mock_stderr: Mocked stderr

    Returns:
        UIManager: Test UI manager
    """
    return UIManager()


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
)
@pytest.mark.asyncio
async def test_get_input(ui_manager: UIManager, mocker: MockerFixture, mock_stderr: MockerFixture) -> None:
    """Test user input handling.

    Args:
        ui_manager: Test UI manager
        mocker: Pytest mocker fixture
        mock_stderr: Mocked stderr
    """
    # Test with custom prompt
    mock_input = mocker.patch("asyncio.to_thread", return_value="test input")
    result = await ui_manager.get_input("Test prompt: ")

    # Verify stderr.flush was called
    assert mock_stderr.flush.call_count > 0, "stderr.flush() should be called at least once"
    mock_input.assert_called_once_with(input, "Test prompt: ")
    assert result == "test input", "Input value should match mock return value"

    # Reset mocks
    mock_stderr.flush.reset_mock()
    mock_input.reset_mock()

    # Test with default prompt
    result = await ui_manager.get_input()
    assert mock_stderr.flush.call_count > 0, "stderr.flush() should be called at least once"
    mock_input.assert_called_once_with(input, "You: ")
    assert result == "test input", "Input value should match mock return value"


@pytest.mark.asyncio
async def test_get_input_error(ui_manager: UIManager, mocker: MockerFixture) -> None:
    """Test user input error handling.

    Args:
        ui_manager: Test UI manager
        mocker: Pytest mocker fixture
    """
    mock_logger = mocker.patch.object(ui_manager, "_logger")
    error = ValueError("Test error")
    mocker.patch("asyncio.to_thread", side_effect=error)

    with pytest.raises(ValueError, match="Test error"):
        await ui_manager.get_input()

    # Verify error was logged
    mock_logger.error.assert_called_once_with("Error getting input", error=str(error))


@pytest.mark.asyncio
async def test_display_welcome(ui_manager: UIManager, mocker: MockerFixture) -> None:
    """Test welcome message display.

    Args:
        ui_manager: Test UI manager
        mocker: Pytest mocker fixture
    """
    mock_rprint = mocker.patch("democracy_exe.chatbot.utils.terminal_utils.ui_manager.rprint")
    await ui_manager.display_welcome()
    mock_rprint.assert_called_once_with(
        "[bold green]Welcome to the DemocracyExeAI Chatbot! Type 'quit' to exit.[/bold green]"
    )


@pytest.mark.asyncio
async def test_display_goodbye(ui_manager: UIManager, mocker: MockerFixture) -> None:
    """Test goodbye message display.

    Args:
        ui_manager: Test UI manager
        mocker: Pytest mocker fixture
    """
    mock_rprint = mocker.patch("democracy_exe.chatbot.utils.terminal_utils.ui_manager.rprint")
    await ui_manager.display_goodbye()
    mock_rprint.assert_called_once_with("[bold red]Goodbye![/bold red]")


@pytest.mark.asyncio
async def test_display_error(ui_manager: UIManager, mocker: MockerFixture) -> None:
    """Test error message display.

    Args:
        ui_manager: Test UI manager
        mocker: Pytest mocker fixture
    """
    mock_rprint = mocker.patch("democracy_exe.chatbot.utils.terminal_utils.ui_manager.rprint")
    error_message = "Test error message"
    await ui_manager.display_error(error_message)
    mock_rprint.assert_called_once_with(f"[bold red]{error_message}[/bold red]")


@pytest.mark.asyncio
async def test_display_response(ui_manager: UIManager, mocker: MockerFixture) -> None:
    """Test response message display.

    Args:
        ui_manager: Test UI manager
        mocker: Pytest mocker fixture
    """
    mock_rprint = mocker.patch("democracy_exe.chatbot.utils.terminal_utils.ui_manager.rprint")
    response = "Test response"
    await ui_manager.display_response(response)
    mock_rprint.assert_called_once_with(f"[bold blue]AI:[/bold blue] {response}")


@pytest.mark.asyncio
async def test_ui_manager_context(ui_manager: UIManager, mock_stderr: MockerFixture) -> None:
    """Test UI manager context management.

    Args:
        ui_manager: Test UI manager
        mock_stderr: Mocked stderr
    """
    async with ui_manager:
        pass
    mock_stderr.flush.assert_called_once()


@pytest.mark.asyncio
async def test_flushing_stderr(ui_manager: UIManager) -> None:
    """Test stderr flushing.

    Args:
        ui_manager: Test UI manager
    """
    message = "Test message"
    ui_manager._stderr_handler.write(message)
    # Note: We can't easily test the actual flushing in a unit test
    # but we can verify the handler exists and accepts writes
