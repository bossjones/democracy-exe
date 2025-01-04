"""Tests for UpdateFileTool."""

from __future__ import annotations

import os
import pathlib

from typing import TYPE_CHECKING

import pytest

from pytest_mock import MockerFixture

from democracy_exe.agentic.tools.update_file_tool import UpdateFileResponse, UpdateFileTool


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def update_file_tool() -> UpdateFileTool:
    """Create UpdateFileTool instance for testing.

    Returns:
        UpdateFileTool instance
    """
    return UpdateFileTool()


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


@pytest.mark.toolonly
def test_validate_path(update_file_tool: UpdateFileTool, test_dir: pathlib.Path) -> None:
    """Test path validation logic.

    Args:
        update_file_tool: UpdateFileTool instance
        test_dir: Temporary test directory
    """
    # Create test file
    test_file = test_dir / "existing.txt"
    test_file.write_text("existing content")

    # Test valid path with existing file
    dir_path, file_path = update_file_tool._validate_path(str(test_dir), "existing.txt")
    assert dir_path == test_dir
    assert file_path == str(test_dir / "existing.txt")

    # Test with non-existent file
    with pytest.raises(ValueError, match="File does not exist"):
        update_file_tool._validate_path(str(test_dir), "nonexistent.txt")

    # Test with invalid directory
    with pytest.raises(ValueError, match="Directory does not exist"):
        update_file_tool._validate_path("/nonexistent/dir", "test.txt")


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_success(
    update_file_tool: UpdateFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """Test successful asynchronous file update.

    Args:
        update_file_tool: UpdateFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    # Create test file
    test_file = test_dir / "test.txt"
    test_file.write_text("Original content")

    result = await update_file_tool.arun({
        "file_name": "test.txt",
        "content": "Updated content",
        "directory": str(test_dir),
    })

    # Verify response
    assert result["status"] == "success"
    assert result["file_path"] == str(test_dir / "test.txt")
    assert result.get("error") is None

    # Verify file was updated with correct content
    assert test_file.read_text() == "Updated content"

    # Verify logging
    assert "Starting asynchronous file update" in caplog.text
    assert "File update completed successfully" in caplog.text


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_nonexistent_file(
    update_file_tool: UpdateFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """Test file update when file doesn't exist.

    Args:
        update_file_tool: UpdateFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    result = await update_file_tool.arun({
        "file_name": "nonexistent.txt",
        "content": "New content",
        "directory": str(test_dir),
    })

    # Verify error response
    assert result["status"] == "error"
    assert "File does not exist" in result["error"]
    assert result["file_path"] == ""

    # Verify file was not created
    assert not (test_dir / "nonexistent.txt").exists()

    # Verify logging
    assert "File does not exist" in caplog.text


@pytest.mark.toolonly
def test_run_success(update_file_tool: UpdateFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture) -> None:
    """Test successful synchronous file update.

    Args:
        update_file_tool: UpdateFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    # Create test file
    test_file = test_dir / "test.txt"
    test_file.write_text("Original content")

    result = update_file_tool.run({"file_name": "test.txt", "content": "Updated content", "directory": str(test_dir)})

    # Verify response
    assert result["status"] == "success"
    assert result["file_path"] == str(test_dir / "test.txt")
    assert result.get("error") is None

    # Verify file was updated with correct content
    assert test_file.read_text() == "Updated content"

    # Verify logging
    assert "Starting synchronous file update" in caplog.text
    assert "File update completed successfully" in caplog.text


@pytest.mark.toolonly
def test_run_with_default_directory(
    update_file_tool: UpdateFileTool, caplog: LogCaptureFixture, tmp_path: pathlib.Path, monkeypatch: MonkeyPatch
) -> None:
    """Test file update with default directory.

    Args:
        update_file_tool: UpdateFileTool instance
        caplog: Pytest fixture for capturing log messages
        tmp_path: Pytest fixture providing temporary directory
        monkeypatch: Pytest fixture for modifying environment
    """
    # Change working directory to tmp_path
    monkeypatch.chdir(tmp_path)

    # Create scratchpad directory and test file
    scratchpad_dir = tmp_path / "scratchpad"
    scratchpad_dir.mkdir()
    test_file = scratchpad_dir / "test.txt"
    test_file.write_text("Original content")

    result = update_file_tool.run({"file_name": "test.txt", "content": "Updated content"})

    # Verify response
    assert result["status"] == "success"
    assert result["file_path"] == str(scratchpad_dir / "test.txt")

    # Verify file was updated
    assert test_file.read_text() == "Updated content"

    # Verify logging
    assert "Starting synchronous file update" in caplog.text
    assert " Asynchronous file update completed successfully" in caplog.text


@pytest.mark.toolonly
def test_run_permission_error(
    update_file_tool: UpdateFileTool, test_dir: pathlib.Path, mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    """Test handling of permission errors during file update.

    Args:
        update_file_tool: UpdateFileTool instance
        test_dir: Temporary test directory
        mocker: Pytest mocker fixture
        caplog: Pytest fixture for capturing log messages
    """
    # Create test file
    test_file = test_dir / "test.txt"
    test_file.write_text("Original content")

    # Mock open to raise PermissionError
    mocker.patch("builtins.open", side_effect=PermissionError("Permission denied"))

    result = update_file_tool.run({"file_name": "test.txt", "content": "Updated content", "directory": str(test_dir)})

    # Verify error response
    assert result["status"] == "error"
    assert "Permission denied" in result["error"]
    assert result["file_path"] == ""

    # Verify file was not modified
    assert test_file.read_text() == "Original content"

    # Verify logging
    assert "File update failed" in caplog.text
    assert "Permission denied" in caplog.text


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_io_error(
    update_file_tool: UpdateFileTool, test_dir: pathlib.Path, mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    """Test handling of IO errors during async file update.

    Args:
        update_file_tool: UpdateFileTool instance
        test_dir: Temporary test directory
        mocker: Pytest mocker fixture
        caplog: Pytest fixture for capturing log messages
    """
    # Create test file
    test_file = test_dir / "test.txt"
    test_file.write_text("Original content")

    # Mock aiofiles.open to raise IOError
    mocker.patch("aiofiles.open", side_effect=OSError("Disk full"))

    result = await update_file_tool.arun({
        "file_name": "test.txt",
        "content": "Updated content",
        "directory": str(test_dir),
    })

    # Verify error response
    assert result["status"] == "error"
    assert "Disk full" in result["error"]
    assert result["file_path"] == ""

    # Verify file was not modified
    assert test_file.read_text() == "Original content"

    # Verify logging
    assert "Asynchronous file update failed" in caplog.text
    assert "Disk full" in caplog.text
