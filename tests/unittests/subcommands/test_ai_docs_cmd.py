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
from unittest.mock import AsyncMock, MagicMock

from typer.testing import CliRunner, Result

import pytest

from democracy_exe.asynctyper_testing import AsyncCliRunner
from democracy_exe.subcommands.ai_docs_cmd import APP
from democracy_exe.utils.ai_docs_utils.extract_repo import extract_local_directory
from democracy_exe.utils.ai_docs_utils.generate_docs import generate_docs_from_local_repo


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
) -> None:
    """Test the generate command.

    Args:
        runner: CLI test runner
        mock_repo_directory: Mock repository directory
        monkeypatch: Pytest monkeypatch fixture
    """
    mock_generate = AsyncMock(return_value="Generated docs")
    monkeypatch.setattr(
        "democracy_exe.utils.ai_docs_utils.generate_docs.generate_docs_from_local_repo",
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
) -> None:
    """Test the generate-async command.

    Args:
        async_runner: Async CLI test runner
        mock_repo_directory: Mock repository directory
        monkeypatch: Pytest monkeypatch fixture
    """
    mock_generate = AsyncMock(return_value="Generated docs")
    monkeypatch.setattr(
        "democracy_exe.utils.ai_docs_utils.generate_docs.generate_docs_from_local_repo",
        mock_generate,
    )

    result = async_runner.invoke(APP, ["generate-async", str(mock_repo_directory)])
    assert result.exit_code == 0
    assert "Documentation generated" in result.stdout


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
