"""Tests for AI docs generation commands.

<test_pattern_documentation>
    <title>Async/Sync Testing Pattern</title>

    <key_insight>
        We do not need separate *_aio_* tests for async commands because all commands ultimately run async code.
    </key_insight>

    <technical_reasons>
        <reason>All commands (both sync and async) ultimately run async code</reason>
        <reason>The setup_event_loop fixture ensures all tests have an event loop available</reason>
        <reason>The sync commands wrap async operations in run_until_complete</reason>
        <reason>AsyncTyperImproved handles the async-to-sync conversion internally</reason>
    </technical_reasons>

    <conclusion>
        Testing the sync command interface provides full coverage of the async functionality.
    </conclusion>

    <implementation_note>
        When adding new tests to this module, do not create redundant *_aio_* versions of tests.
        Instead, ensure the sync version properly tests the async functionality through the event loop.
    </implementation_note>
</test_pattern_documentation>
"""

# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import asyncio
import datetime
import os
import shutil

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


@pytest.fixture(autouse=True, scope="function")
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


def test_cli_generate_module_docs_recursive(
    mock_nested_module_dir: Path,
    mocker: MockerFixture,
    runner: CliRunner,
) -> None:
    """Test recursive documentation generation.

    Args:
        mock_nested_module_dir: Path to test directory with nested structure
        mocker: Pytest mocker fixture
        runner: CLI test runner
    """
    # Mock the generate_module_docs function
    mock_generate = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.generate_module_docs")
    mock_confirm = mocker.patch("typer.confirm", return_value=True)

    # Run the command with recursive flag
    result = runner.invoke(APP, ["generate-module", str(mock_nested_module_dir), "--recursive"])

    # Verify success
    assert result.exit_code == 0

    # Verify files were found (4 Python files total)
    assert "Found Python files:" in result.stdout
    assert "root.py" in result.stdout
    assert "module1.py" in result.stdout
    assert "module2.py" in result.stdout
    assert "deep_module.py" in result.stdout

    # Verify user confirmation was requested
    mock_confirm.assert_called_once()

    # Verify generate_module_docs was called for each file
    assert mock_generate.call_count == 4
    mock_generate.assert_has_calls(
        [
            mocker.call(str(mock_nested_module_dir / "root.py")),
            mocker.call(str(mock_nested_module_dir / "subdir" / "module1.py")),
            mocker.call(str(mock_nested_module_dir / "subdir" / "module2.py")),
            mocker.call(str(mock_nested_module_dir / "subdir" / "deepdir" / "deep_module.py")),
        ],
        any_order=True,
    )


def test_cli_generate_module_docs_recursive_with_force(
    mock_nested_module_dir: Path,
    mocker: MockerFixture,
    runner: CliRunner,
) -> None:
    """Test recursive documentation generation with force flag.

    Args:
        mock_nested_module_dir: Path to test directory with nested structure
        mocker: Pytest mocker fixture
        runner: CLI test runner
    """
    # Mock the generate_module_docs function
    mock_generate = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.generate_module_docs")
    mock_confirm = mocker.patch("typer.confirm")

    # Run the command with recursive and force flags
    result = runner.invoke(APP, ["generate-module", str(mock_nested_module_dir), "--recursive", "--force"])

    # Verify success
    assert result.exit_code == 0

    # Verify files were found
    assert "Found Python files:" in result.stdout
    assert "Forcing documentation generation for all files..." in result.stdout

    # Verify user confirmation was NOT requested
    mock_confirm.assert_not_called()

    # Verify generate_module_docs was called for each file
    assert mock_generate.call_count == 4


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
def test_cli_generate_module_docs_recursive_user_cancel(
    mock_nested_module_dir: Path,
    mocker: MockerFixture,
    runner: CliRunner,
) -> None:
    """Test recursive documentation generation when user cancels.

    Args:
        mock_nested_module_dir: Path to test directory with nested structure
        mocker: Pytest mocker fixture
        runner: CLI test runner
    """
    # Mock the generate_module_docs function
    mock_generate = mocker.patch("democracy_exe.subcommands.ai_docs_cmd.generate_module_docs")
    mock_confirm = mocker.patch("typer.confirm", return_value=False)

    # Run the command with recursive flag
    result = runner.invoke(APP, ["generate-module", str(mock_nested_module_dir), "--recursive"])

    # Verify operation was cancelled
    assert result.exit_code == 0
    assert "Operation cancelled" in result.stdout

    # Verify user confirmation was requested
    mock_confirm.assert_called_once()

    # Verify generate_module_docs was NOT called
    mock_generate.assert_not_called()


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
def test_cli_generate_module_docs_recursive_with_errors(
    mock_nested_module_dir: Path,
    mocker: MockerFixture,
    runner: CliRunner,
) -> None:
    """Test recursive documentation generation with errors.

    Args:
        mock_nested_module_dir: Path to test directory with nested structure
        mocker: Pytest mocker fixture
        runner: CLI test runner
    """

    # Mock generate_module_docs to succeed for first file and fail for second
    def mock_generate_side_effect(path: str) -> None:
        if "module1.py" in path:
            raise Exception("Test error")
        return None

    mock_generate = mocker.patch(
        "democracy_exe.subcommands.ai_docs_cmd.generate_module_docs", side_effect=mock_generate_side_effect
    )
    mock_confirm = mocker.patch("typer.confirm", side_effect=[True, True])  # Continue after error

    # Run the command with recursive flag
    result = runner.invoke(APP, ["generate-module", str(mock_nested_module_dir), "--recursive"])

    # Verify completion despite errors
    assert result.exit_code == 0
    assert "Error processing" in result.stdout
    assert "Test error" in result.stdout
    assert "Documentation generation complete!" in result.stdout

    # Verify user was prompted to continue after error
    assert mock_confirm.call_count == 2  # Initial confirmation + error continuation

    # Verify generate_module_docs was called for all files
    assert mock_generate.call_count == 4
