"""Tests for AI docs generation commands"""

from __future__ import annotations

import asyncio
import os

from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from typer.testing import CliRunner

import pytest

from democracy_exe.subcommands.ai_docs_cmd import APP


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

logger = structlog.get_logger(__name__)


@pytest.fixture
def runner() -> Generator[CliRunner, None, None]:
    """Get a Typer CLI test runner.

    Returns:
        Generator[CliRunner, None, None]: CLI runner instance
    """
    yield CliRunner()


@pytest.fixture
def mock_repo_directory(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a mock repository directory structure.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Generator[Path, None, None]: Path to mock repository
    """
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()

    # Create some mock files
    (repo_dir / "main.py").write_text("def main(): pass")
    (repo_dir / "README.md").write_text("# Mock Repo")

    yield repo_dir


@pytest.fixture
def mock_docs_output(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a mock docs output directory.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Generator[Path, None, None]: Path to mock docs directory
    """
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    yield docs_dir


@pytest.fixture
def mock_extract_output(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a mock extract output directory.

    Args:
        tmp_path: Pytest temporary path fixture

    Returns:
        Generator[Path, None, None]: Path to mock extract directory
    """
    extract_dir = tmp_path / "extract"
    extract_dir.mkdir()
    yield extract_dir


@pytest.fixture(autouse=True)
def setup_event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create and set a new event loop for each test.

    Returns:
        Generator[asyncio.AbstractEventLoop, None, None]: Event loop for test
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


def test_cli_generate_docs(runner: CliRunner, mock_repo_directory: Path) -> None:
    """Test the generate docs command with valid input.

    Args:
        runner: Typer CLI test runner
        mock_repo_directory: Mock repository directory
    """
    result = runner.invoke(APP, ["generate", str(mock_repo_directory)])
    assert result.exit_code == 0
    assert "Documentation generated" in result.stdout


def test_cli_generate_docs_invalid_path(runner: CliRunner) -> None:
    """Test the generate docs command with invalid path.

    Args:
        runner: Typer CLI test runner
    """
    result = runner.invoke(APP, ["generate", "/nonexistent/path"])
    assert result.exit_code == 1
    assert "Directory does not exist" in result.stdout


def test_cli_extract_repo(runner: CliRunner, mock_repo_directory: Path) -> None:
    """Test the extract repo command with valid input.

    Args:
        runner: Typer CLI test runner
        mock_repo_directory: Mock repository directory
    """
    result = runner.invoke(APP, ["extract", str(mock_repo_directory)])
    assert result.exit_code == 0
    assert "Repository extracted" in result.stdout


def test_cli_extract_repo_invalid_path(runner: CliRunner) -> None:
    """Test the extract repo command with invalid path.

    Args:
        runner: Typer CLI test runner
    """
    result = runner.invoke(APP, ["extract", "/nonexistent/path"])
    assert result.exit_code == 1
    assert "Directory does not exist" in result.stdout


@pytest.mark.asyncio
async def test_aio_cli_generate_docs(runner: CliRunner, mock_repo_directory: Path) -> None:
    """Test the async generate docs command with valid input.

    Args:
        runner: Typer CLI test runner
        mock_repo_directory: Mock repository directory
    """
    result = runner.invoke(APP, ["generate-async", str(mock_repo_directory)])
    assert result.exit_code == 0
    assert "Documentation generated" in result.stdout


@pytest.mark.asyncio
async def test_aio_cli_generate_docs_invalid_path(runner: CliRunner) -> None:
    """Test the async generate docs command with invalid path.

    Args:
        runner: Typer CLI test runner
    """
    result = runner.invoke(APP, ["generate-async", "/nonexistent/path"])
    assert result.exit_code == 1
    assert "Directory does not exist" in result.stdout


@pytest.mark.asyncio
async def test_aio_cli_extract_repo(runner: CliRunner, mock_repo_directory: Path) -> None:
    """Test the async extract repo command with valid input.

    Args:
        runner: Typer CLI test runner
        mock_repo_directory: Mock repository directory
    """
    result = runner.invoke(APP, ["extract-async", str(mock_repo_directory)])
    assert result.exit_code == 0
    assert "Repository extracted" in result.stdout


@pytest.mark.asyncio
async def test_aio_cli_extract_repo_invalid_path(runner: CliRunner) -> None:
    """Test the async extract repo command with invalid path.

    Args:
        runner: Typer CLI test runner
    """
    result = runner.invoke(APP, ["extract-async", "/nonexistent/path"])
    assert result.exit_code == 1
    assert "Directory does not exist" in result.stdout
