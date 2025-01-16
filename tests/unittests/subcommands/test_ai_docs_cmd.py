"""Tests for AI docs generation commands"""

# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import asyncio
import os

from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
import typer

from typer.testing import CliRunner, Result

import pytest

from democracy_exe.asynctyper_testing import AsyncCliRunner
from democracy_exe.subcommands.ai_docs_cmd import APP, cli_generate_module_docs
from democracy_exe.utils.ai_docs_utils.extract_repo import extract_local_directory
from democracy_exe.utils.ai_docs_utils.generate_docs import agenerate_docs_from_local_repo


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI runner for testing.

    Returns:
        CliRunner: The test runner
    """
    return CliRunner()


@pytest.fixture
def async_runner() -> AsyncCliRunner:
    """Create an async CLI runner for testing.

    Returns:
        AsyncCliRunner: The async test runner
    """
    return AsyncCliRunner()


@pytest.fixture
def mock_repo_directory(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a mock repository directory for testing.

    Args:
        tmp_path: Pytest temporary path fixture

    Yields:
        Path: Path to the mock repository
    """
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    yield repo_dir


@pytest.fixture
def mock_docs_output() -> str:
    """Create mock documentation output.

    Returns:
        str: Mock documentation string
    """
    return "Generated docs"


@pytest.fixture
def mock_extract_output() -> str:
    """Create mock extraction output.

    Returns:
        str: Mock extraction string
    """
    return "output.txt"


@pytest.fixture(autouse=True)
def setup_event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create and set a new event loop for each test.

    Yields:
        asyncio.AbstractEventLoop: The event loop for testing
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


def test_cli_generate_docs(
    runner: CliRunner,
    mock_repo_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Test the generate command.

    Args:
        runner: CLI test runner
        mock_repo_directory: Mock repository directory
        monkeypatch: Pytest monkeypatch fixture
        mocker: MockerFixture
    """
    mock_generate = mocker.AsyncMock(return_value="Generated docs")
    monkeypatch.setattr(
        "democracy_exe.utils.ai_docs_utils.generate_docs.agenerate_docs_from_local_repo",
        mock_generate,
    )

    result = runner.invoke(APP, ["generate", str(mock_repo_directory)])
    assert result.exit_code == 0
    assert "Documentation generated" in result.stdout


def test_cli_generate_docs_invalid_path(runner: CliRunner) -> None:
    """Test generate command with invalid path.

    Args:
        runner: CLI test runner
    """
    result = runner.invoke(APP, ["generate", "invalid/path"])
    assert result.exit_code == 1
    assert "Directory does not exist" in result.stdout


def test_cli_extract_repo(
    runner: CliRunner,
    mock_repo_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the extract command.

    Args:
        runner: CLI test runner
        mock_repo_directory: Mock repository directory
        monkeypatch: Pytest monkeypatch fixture
    """

    def mock_extract(repo_dir: str) -> str:
        return "output.txt"

    monkeypatch.setattr(
        "democracy_exe.utils.ai_docs_utils.extract_repo.extract_local_directory",
        mock_extract,
    )

    result = runner.invoke(APP, ["extract", str(mock_repo_directory)])
    assert result.exit_code == 0
    assert "Repository extracted" in result.stdout


def test_cli_extract_repo_invalid_path(runner: CliRunner) -> None:
    """Test extract command with invalid path.

    Args:
        runner: CLI test runner
    """
    result = runner.invoke(APP, ["extract", "invalid/path"])
    assert result.exit_code == 1
    assert "Directory does not exist" in result.stdout


@pytest.mark.asyncio
async def test_aio_cli_generate_docs(
    async_runner: AsyncCliRunner,
    mock_repo_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
) -> None:
    """Test the generate-async command.

    Args:
        async_runner: Async CLI test runner
        mock_repo_directory: Mock repository directory
        monkeypatch: Pytest monkeypatch fixture
    """
    mock_generate = mocker.AsyncMock(return_value="Generated docs")
    monkeypatch.setattr(
        "democracy_exe.utils.ai_docs_utils.generate_docs.agenerate_docs_from_local_repo",
        mock_generate,
    )

    # result = async_runner.invoke(APP, ["generate-async", str(mock_repo_directory)])
    # assert result.exit_code == 0
    # assert "Documentation generated" in result.stdout

    # Use asyncio.create_task to ensure the coroutine is properly scheduled
    result = await asyncio.create_task(async_runner.invoke(APP, ["generate-async", str(mock_repo_directory)]))

    assert result.exit_code == 0
    assert "Documentation generated" in result.stdout
    # Verify the mock was called
    mock_generate.assert_called_once()


@pytest.mark.asyncio
async def test_aio_cli_generate_docs_invalid_path(async_runner: AsyncCliRunner) -> None:
    """Test generate-async command with invalid path.

    Args:
        async_runner: Async CLI test runner
    """
    result = async_runner.invoke(APP, ["generate-async", "invalid/path"])
    assert result.exit_code == 1
    assert "Directory does not exist" in result.stdout


@pytest.mark.asyncio
async def test_aio_cli_extract_repo(
    async_runner: AsyncCliRunner,
    mock_repo_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test the extract-async command.

    Args:
        async_runner: Async CLI test runner
        mock_repo_directory: Mock repository directory
        monkeypatch: Pytest monkeypatch fixture
    """

    def mock_extract(repo_dir: str) -> str:
        return "output.txt"

    monkeypatch.setattr(
        "democracy_exe.utils.ai_docs_utils.extract_repo.extract_local_directory",
        mock_extract,
    )

    result = async_runner.invoke(APP, ["extract-async", str(mock_repo_directory)])
    assert result.exit_code == 0
    assert "Repository extracted" in result.stdout


@pytest.mark.asyncio
async def test_aio_cli_extract_repo_invalid_path(async_runner: AsyncCliRunner) -> None:
    """Test extract-async command with invalid path.

    Args:
        async_runner: Async CLI test runner
    """
    result = async_runner.invoke(APP, ["extract-async", "invalid/path"])
    assert result.exit_code == 1
    assert "Directory does not exist" in result.stdout


@pytest.fixture
def mock_module_file(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary Python module file for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Yields:
        Path to temporary Python file
    """
    file_path = tmp_path / "test_module.py"
    file_path.write_text("def test_function():\n    pass\n")
    yield file_path
    if file_path.exists():
        file_path.unlink()


@pytest.fixture
def mock_module_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary directory with Python modules for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Yields:
        Path to temporary directory
    """
    dir_path = tmp_path / "test_module_dir"
    dir_path.mkdir()
    (dir_path / "module1.py").write_text("def test_function1():\n    pass\n")
    (dir_path / "module2.py").write_text("def test_function2():\n    pass\n")
    yield dir_path
    if dir_path.exists():
        for file in dir_path.glob("*.py"):
            file.unlink()
        dir_path.rmdir()


def test_cli_generate_module_docs_file(
    mock_module_file: Path, mocker: MockerFixture, capsys: CaptureFixture[str]
) -> None:
    """Test generating docs for a Python file.

    Args:
        mock_module_file: Path to test Python file
        mocker: Pytest mocker fixture
        capsys: Pytest capture fixture
    """
    # Mock the generate_module_docs function
    mock_generate = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.generate_module_docs")
    mock_rprint = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.rprint")

    # Run the command
    cli_generate_module_docs(str(mock_module_file))

    # Verify generate_module_docs was called with correct path
    mock_generate.assert_called_once_with(str(mock_module_file))

    # Verify success message was printed
    mock_rprint.assert_called_with(f"[green]Module documentation generated for {mock_module_file}[/green]")


def test_cli_generate_module_docs_directory(
    mock_module_dir: Path, mocker: MockerFixture, capsys: CaptureFixture[str]
) -> None:
    """Test generating docs for a directory of Python files.

    Args:
        mock_module_dir: Path to test directory
        mocker: Pytest mocker fixture
        capsys: Pytest capture fixture
    """
    # Mock the generate_module_docs function
    mock_generate = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.generate_module_docs")
    mock_rprint = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.rprint")

    # Run the command
    cli_generate_module_docs(str(mock_module_dir))

    # Verify generate_module_docs was called with correct path
    mock_generate.assert_called_once_with(str(mock_module_dir))

    # Verify success message was printed
    mock_rprint.assert_called_with(f"[green]Module documentation generated for {mock_module_dir}[/green]")


def test_cli_generate_module_docs_nonexistent_path(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test handling of nonexistent path.

    Args:
        tmp_path: Pytest temporary directory fixture
        mocker: Pytest mocker fixture
    """
    nonexistent_path = tmp_path / "nonexistent.py"
    mock_rprint = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.rprint")

    # Verify it raises typer.Exit
    with pytest.raises(typer.Exit):
        cli_generate_module_docs(str(nonexistent_path))

    # Verify error message was printed
    mock_rprint.assert_called_with(f"[red]Path does not exist: {nonexistent_path}[/red]")


def test_cli_generate_module_docs_invalid_file_type(tmp_path: Path, mocker: MockerFixture) -> None:
    """Test handling of invalid file type.

    Args:
        tmp_path: Pytest temporary directory fixture
        mocker: Pytest mocker fixture
    """
    invalid_file = tmp_path / "test.txt"
    invalid_file.write_text("test content")
    mock_rprint = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.rprint")

    # Verify it raises typer.Exit
    with pytest.raises(typer.Exit):
        cli_generate_module_docs(str(invalid_file))

    # Verify error message was printed
    mock_rprint.assert_called_with(f"[red]Path must be a Python file or directory: {invalid_file}[/red]")


def test_cli_generate_module_docs_generation_error(
    mock_module_file: Path,
    mocker: MockerFixture,
) -> None:
    """Test handling of documentation generation error.

    Args:
        mock_module_file: Path to test Python file
        mocker: Pytest mocker fixture
    """
    # Mock generate_module_docs to raise an exception
    mock_generate = mocker.patch(
        "democracy_exe.subcommands.ai_docs_cmd.generate_module_docs", side_effect=Exception("Test error")
    )
    mock_rprint = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.rprint")

    # Use structlog's testing context manager
    with structlog.testing.capture_logs() as captured:
        # Verify it raises typer.Exit
        with pytest.raises(typer.Exit):
            cli_generate_module_docs(str(mock_module_file))

        # Verify error was logged
        assert any("Error generating module documentation" in log.get("event", "") for log in captured)
        assert any("Test error" in str(log.get("error", "")) for log in captured)

    # Verify error message was printed
    mock_rprint.assert_called_with("[red]Error: Test error[/red]")
