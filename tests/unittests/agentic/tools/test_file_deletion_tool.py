"""Tests for DeleteFileTool."""

from __future__ import annotations

import os
import pathlib

from typing import TYPE_CHECKING

import structlog

from structlog.testing import capture_logs

import pytest

from pytest_mock import MockerFixture

from democracy_exe.agentic.tools.delete_file_tool import DeleteFileResponse, DeleteFileTool


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture

logger = structlog.get_logger(__name__)


@pytest.fixture
def delete_file_tool() -> DeleteFileTool:
    """Create DeleteFileTool instance for testing.

    Returns:
        DeleteFileTool instance
    """
    return DeleteFileTool()


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
def test_validate_path(delete_file_tool: DeleteFileTool, test_dir: pathlib.Path) -> None:
    """Test path validation logic.

    Args:
        delete_file_tool: DeleteFileTool instance
        test_dir: Temporary test directory
    """
    # Create test file
    test_file = test_dir / "existing.txt"
    test_file.write_text("test content")

    # Test valid path with existing file
    dir_path, file_path = delete_file_tool._validate_path(str(test_dir), "existing.txt")
    assert dir_path == test_dir
    assert file_path == str(test_dir / "existing.txt")

    # Test with non-existent file
    with pytest.raises(ValueError, match="File does not exist"):
        delete_file_tool._validate_path(str(test_dir), "nonexistent.txt")

    # Test with invalid directory
    with pytest.raises(ValueError, match="Directory does not exist"):
        delete_file_tool._validate_path("/nonexistent/dir", "test.txt")


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_success(
    delete_file_tool: DeleteFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """Test successful asynchronous file deletion.

    Args:
        delete_file_tool: DeleteFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        # Create test file
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")

        result = await delete_file_tool.arun({"file_name": "test.txt", "directory": str(test_dir), "force": True})

        # Verify response
        assert result["status"] == "success"
        assert result["file_path"] == str(test_dir / "test.txt")
        assert result.get("error") is None

        # Verify file was deleted
        assert not test_file.exists()

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_success:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event").startswith("Starting asynchronous file deletion for test.txt") for log in captured
        ), "Expected 'Starting asynchronous file deletion' message not found in logs"

        assert any(log.get("event") == "Asynchronous file deletion completed successfully" for log in captured), (
            "Expected 'Asynchronous file deletion completed successfully' message not found in logs"
        )


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_without_force(
    delete_file_tool: DeleteFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """Test file deletion without force flag.

    Args:
        delete_file_tool: DeleteFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        # Create test file
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")

        result = await delete_file_tool.arun({"file_name": "test.txt", "directory": str(test_dir), "force": False})

        # Verify response requires confirmation
        assert result["status"] == "confirmation_required"
        assert result["file_path"] == str(test_dir / "test.txt")
        assert result["requires_confirmation"] is True

        # Verify file was not deleted
        assert test_file.exists()
        assert test_file.read_text() == "test content"

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_without_force:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Force flag not set, requiring confirmation" for log in captured), (
            "Expected 'Force flag not set, requiring confirmation' message not found in logs"
        )


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_nonexistent_file(
    delete_file_tool: DeleteFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture
) -> None:
    """Test file deletion when file doesn't exist.

    Args:
        delete_file_tool: DeleteFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        result = await delete_file_tool.arun({
            "file_name": "nonexistent.txt",
            "directory": str(test_dir),
            "force": True,
        })

        # Verify error response
        assert result["status"] == "error"
        assert "File does not exist" in result["error"]
        assert result["file_path"] == ""

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_nonexistent_file:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event").startswith("Path validation failed: File does not exist:") for log in captured), (
            "Expected 'Path validation failed: File does not exist' message not found in logs"
        )


@pytest.mark.toolonly
def test_run_success(delete_file_tool: DeleteFileTool, test_dir: pathlib.Path, caplog: LogCaptureFixture) -> None:
    """Test successful synchronous file deletion.

    Args:
        delete_file_tool: DeleteFileTool instance
        test_dir: Temporary test directory
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        # Create test file
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")

        result = delete_file_tool.run({"file_name": "test.txt", "directory": str(test_dir), "force": True})

        # Verify response
        assert result["status"] == "success"
        assert result["file_path"] == str(test_dir / "test.txt")
        assert result.get("error") is None

        # Verify file was deleted
        assert not test_file.exists()

        # Debug: Print captured logs
        print("\nCaptured logs in test_run_success:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event").startswith("Starting synchronous file deletion for test.txt") for log in captured
        ), "Expected 'Starting synchronous file deletion' message not found in logs"

        assert any(log.get("event") == "File deletion completed successfully" for log in captured), (
            "Expected 'File deletion completed successfully' message not found in logs"
        )


@pytest.mark.toolonly
def test_run_with_default_directory(
    delete_file_tool: DeleteFileTool, caplog: LogCaptureFixture, tmp_path: pathlib.Path, monkeypatch: MonkeyPatch
) -> None:
    """Test file deletion with default directory.

    Args:
        delete_file_tool: DeleteFileTool instance
        caplog: Pytest fixture for capturing log messages
        tmp_path: Pytest fixture providing temporary directory
        monkeypatch: Pytest fixture for modifying environment
    """
    with capture_logs() as captured:
        # Change working directory to tmp_path
        monkeypatch.chdir(tmp_path)

        # Create scratchpad directory and test file
        scratchpad_dir = tmp_path / "scratchpad"
        scratchpad_dir.mkdir()
        test_file = scratchpad_dir / "test.txt"
        test_file.write_text("test content")

        result = delete_file_tool.run({"file_name": "test.txt", "force": True})

        # Verify response
        assert result["status"] == "success"
        assert result["file_path"] == str(scratchpad_dir / "test.txt")

        # Verify file was deleted
        assert not test_file.exists()

        # Debug: Print captured logs
        print("\nCaptured logs in test_run_with_default_directory:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event").startswith("Starting synchronous file deletion for test.txt") for log in captured
        ), "Expected 'Starting synchronous file deletion' message not found in logs"

        assert any(log.get("event") == "File deletion completed successfully" for log in captured), (
            "Expected 'File deletion completed successfully' message not found in logs"
        )


@pytest.mark.toolonly
def test_run_permission_error(
    delete_file_tool: DeleteFileTool, test_dir: pathlib.Path, mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    """Test handling of permission errors during file deletion.

    Args:
        delete_file_tool: DeleteFileTool instance
        test_dir: Temporary test directory
        mocker: Pytest mocker fixture
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        # Create test file
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")

        # Mock os.remove to raise PermissionError
        mocker.patch("os.remove", side_effect=PermissionError("Permission denied"))

        result = delete_file_tool.run({"file_name": "test.txt", "directory": str(test_dir), "force": True})

        # Verify error response
        assert result["status"] == "error"
        assert "Permission denied" in result["error"]
        assert result["file_path"] == ""

        # Verify file was not deleted
        assert test_file.exists()
        assert test_file.read_text() == "test content"

        # Debug: Print captured logs
        print("\nCaptured logs in test_run_permission_error:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event").startswith("File deletion failed: Permission denied") for log in captured), (
            "Expected 'File deletion failed' message not found in logs"
        )


@pytest.mark.toolonly
@pytest.mark.asyncio
async def test_arun_io_error(
    delete_file_tool: DeleteFileTool, test_dir: pathlib.Path, mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    """Test handling of IO errors during async file deletion.

    Args:
        delete_file_tool: DeleteFileTool instance
        test_dir: Temporary test directory
        mocker: Pytest mocker fixture
        caplog: Pytest fixture for capturing log messages
    """
    with capture_logs() as captured:
        # Create test file
        test_file = test_dir / "test.txt"
        test_file.write_text("test content")

        # Mock os.remove to raise IOError
        mocker.patch("os.remove", side_effect=OSError("Device busy"))

        result = await delete_file_tool.arun({"file_name": "test.txt", "directory": str(test_dir), "force": True})

        # Verify error response
        assert result["status"] == "error"
        assert "Device busy" in result["error"]
        assert result["file_path"] == ""

        # Verify file was not deleted
        assert test_file.exists()
        assert test_file.read_text() == "test content"

        # Debug: Print captured logs
        print("\nCaptured logs in test_arun_io_error:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event").startswith("Asynchronous file deletion failed: Device busy") for log in captured), (
            "Expected 'Asynchronous file deletion failed' message not found in logs"
        )
