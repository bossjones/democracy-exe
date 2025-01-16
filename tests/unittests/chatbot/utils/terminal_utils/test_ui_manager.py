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

import sys

from collections.abc import Generator
from typing import Any

import pytest

from pytest_mock import MockerFixture

from democracy_exe.chatbot.utils.terminal_utils import UIManager


@pytest.fixture
def ui_manager() -> UIManager:
    """Create a UI manager instance for testing.

    Returns:
        UIManager: Test UI manager
    """
    return UIManager()


@pytest.fixture
def mock_stderr(mocker: MockerFixture) -> Generator[MockerFixture, None, None]:
    """Mock stderr for testing.

    Args:
        mocker: Pytest mocker fixture

    Yields:
        MockerFixture: Mocked stderr
    """
    mock = mocker.patch.object(sys, "stderr")
    yield mock


@pytest.mark.asyncio
async def test_get_input(ui_manager: UIManager, mocker: MockerFixture, mock_stderr: MockerFixture) -> None:
    """Test user input handling.

    Args:
        ui_manager: Test UI manager
        mocker: Pytest mocker fixture
        mock_stderr: Mocked stderr
    """
    mock_input = mocker.patch("asyncio.to_thread", return_value="test input")
    result = await ui_manager.get_input("Test prompt: ")

    mock_stderr.flush.assert_called_once()
    mock_input.assert_called_once_with(input, "Test prompt: ")
    assert result == "test input"


@pytest.mark.asyncio
async def test_get_input_error(ui_manager: UIManager, mocker: MockerFixture) -> None:
    """Test user input error handling.

    Args:
        ui_manager: Test UI manager
        mocker: Pytest mocker fixture
    """
    mocker.patch("asyncio.to_thread", side_effect=ValueError("Test error"))

    with pytest.raises(ValueError, match="Test error"):
        await ui_manager.get_input()


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
