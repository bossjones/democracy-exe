"""Tests for CreateFileTool."""

from __future__ import annotations

import os
import pathlib

from typing import TYPE_CHECKING

import pytest

from pytest_mock import MockerFixture

from democracy_exe.agentic.tools.create_file_tool import CreateFileResponse, CreateFileTool


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def create_file_tool() -> CreateFileTool:
    """Create CreateFileTool instance for testing.

    Returns:
        CreateFileTool instance
    """
    return CreateFileTool()


@pytest.fixture
def test_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary directory for file operations.

    Args:
        tmp_path: Pytest fixture providing temporary directory

    Returns:
        Path to temporary test directory
    """
    test_dir = tmp_path / "test_scratchpad"
    test_dir.mkdir()
    return test_dir


def test_validate_path(create_file_tool: CreateFileTool, test_dir: pathlib.Path) -> None:
    """Test path validation logic.

    Args:
        create_file_tool: CreateFileTool instance
        test_dir: Temporary test directory
    """
    # Test valid path
    dir_path, file_path = create_file_tool._validate_path(str(test_dir), "test.txt")
    assert dir_path == test_dir
    assert file_path == str(test_dir / "test.txt")

    # Test with existing file
    test_file = test_dir / "existing.txt"
    test_file.write_text("existing content")

    with pytest.raises(ValueError, match="File already exists"):
        create_file_tool._validate_path(str(test_dir), "existing.txt")

    # Test with invalid directory
    with pytest.raises(ValueError, match="Path validation failed"):
        create_file_tool._validate_path("/nonexistent/dir", "test.txt")


@pytest.mark.asyncio
@pytest.mark.toolonly
async def test_arun_success(
    create_file_tool: CreateFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """Test successful asynchronous file creation.

    Args:
        create_file_tool: CreateFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    result = await create_file_tool.arun({
        "file_name": "test.txt",
        "content": "Hello, World!",
        "directory": str(test_dir),
    })

    # Verify response
    assert result["status"] == "success"
    assert result["file_path"] == str(test_dir / "test.txt")
    assert result.get("error") is None

    # Verify file was created with correct content
    created_file = test_dir / "test.txt"
    assert created_file.exists()
    assert created_file.read_text() == "Hello, World!"

    # Verify logging
    assert "Starting asynchronous file creation" in caplog.text
    assert "File creation completed successfully" in caplog.text


@pytest.mark.asyncio
@pytest.mark.toolonly
async def test_arun_existing_file(
    create_file_tool: CreateFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """Test file creation when file already exists.

    Args:
        create_file_tool: CreateFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    # Create file first
    test_file = test_dir / "existing.txt"
    test_file.write_text("existing content")

    result = await create_file_tool.arun({
        "file_name": "existing.txt",
        "content": "New content",
        "directory": str(test_dir),
    })

    # Verify error response
    assert result["status"] == "error"
    assert "File already exists" in result["error"]
    assert result["file_path"] == ""

    # Verify original file is unchanged
    assert test_file.read_text() == "existing content"

    # Verify logging
    assert "File already exists" in caplog.text


@pytest.mark.toolonly
def test_run_success(create_file_tool: CreateFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture) -> None:
    """Test successful synchronous file creation.

    Args:
        create_file_tool: CreateFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    result = create_file_tool.run({"file_name": "test.txt", "content": "Hello, World!", "directory": str(test_dir)})

    # Verify response
    assert result["status"] == "success"
    assert result["file_path"] == str(test_dir / "test.txt")
    assert result.get("error") is None

    # Verify file was created with correct content
    created_file = test_dir / "test.txt"
    assert created_file.exists()
    assert created_file.read_text() == "Hello, World!"

    # Verify logging
    assert "Starting synchronous file creation" in caplog.text
    assert "Synchronous file creation completed successfully" in caplog.text


@pytest.mark.toolonly
def test_run_with_default_directory(
    create_file_tool: CreateFileTool, caplog: LogCaptureFixture, tmp_path: pathlib.Path, monkeypatch: MonkeyPatch
) -> None:
    """Test file creation with default directory.

    Args:
        create_file_tool: CreateFileTool instance
        caplog: Pytest fixture for capturing log messages
        tmp_path: Pytest fixture providing temporary directory
        monkeypatch: Pytest fixture for modifying environment
    """
    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)

    result = create_file_tool.run({"file_name": "test.txt", "content": "Hello, World!"})

    # Verify response
    assert result["status"] == "success"
    scratchpad_dir = tmp_path / "scratchpad"
    assert result["file_path"] == str(scratchpad_dir / "test.txt")

    # Verify file was created
    created_file = scratchpad_dir / "test.txt"
    assert created_file.exists()
    assert created_file.read_text() == "Hello, World!"

    # Verify logging
    assert "Starting synchronous file creation" in caplog.text
    assert "Synchronous file creation completed successfully" in caplog.text


@pytest.mark.toolonly
def test_run_permission_error(
    create_file_tool: CreateFileTool, test_dir: pathlib.Path, mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    """Test handling of permission errors during file creation.

    Args:
        create_file_tool: CreateFileTool instance
        test_dir: Temporary test directory
        mocker: Pytest mocker fixture
        caplog: Pytest fixture for capturing log messages
    """
    # Mock open to raise PermissionError
    mocker.patch("builtins.open", side_effect=PermissionError("Permission denied"))

    result = create_file_tool.run({"file_name": "test.txt", "content": "Hello, World!", "directory": str(test_dir)})

    # Verify error response
    assert result["status"] == "error"
    assert "Permission denied" in result["error"]
    assert result["file_path"] == ""

    # Verify file was not created
    assert not (test_dir / "test.txt").exists()

    # Verify logging
    assert "File creation failed" in caplog.text
    assert "Permission denied" in caplog.text


@pytest.mark.asyncio
@pytest.mark.toolonly
async def test_arun_io_error(
    create_file_tool: CreateFileTool, test_dir: pathlib.Path, mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    """Test handling of IO errors during async file creation.

    Args:
        create_file_tool: CreateFileTool instance
        test_dir: Temporary test directory
        mocker: Pytest mocker fixture
        caplog: Pytest fixture for capturing log messages
    """
    # Mock aiofiles.open to raise IOError
    mocker.patch("aiofiles.open", side_effect=OSError("Disk full"))

    result = await create_file_tool.arun({
        "file_name": "test.txt",
        "content": "Hello, World!",
        "directory": str(test_dir),
    })

    # Verify error response
    assert result["status"] == "error"
    assert "Disk full" in result["error"]
    assert result["file_path"] == ""

    # Verify file was not created
    assert not (test_dir / "test.txt").exists()

    # Verify logging
    assert "Asynchronous file creation failed" in caplog.text
    assert "Disk full" in caplog.text
