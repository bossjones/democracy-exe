from __future__ import annotations

import os
import pathlib
import re

from typing import TYPE_CHECKING

import pytest

from democracy_exe.chatbot.utils.discord_utils import aunlink_orig_file, unlink_orig_file


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a temporary test file.

    Args:
        tmp_path: Pytest fixture providing temporary directory path.

    Returns:
        Path to the created temporary file.
    """
    test_file = tmp_path / "test_file.txt"
    test_file.write_text("test content")
    return test_file


def test_unlink_orig_file(tmp_path: pathlib.PosixPath):
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    assert test_file.exists()

    unlink_orig_file(str(test_file))
    assert not test_file.exists()


def test_unlink_orig_file_nonexistent(tmp_path: pathlib.Path) -> None:
    """Test unlink_orig_file with a nonexistent file.

    This test verifies that the function raises FileNotFoundError
    when attempting to delete a nonexistent file.

    Args:
        tmp_path: Pytest fixture providing temporary directory path
    """
    nonexistent_file = tmp_path / "nonexistent.txt"

    with pytest.raises(FileNotFoundError):
        unlink_orig_file(str(nonexistent_file))


@pytest.mark.asyncio
async def test_aunlink_orig_file(tmp_path: pathlib.Path) -> None:
    """Test the async unlink_orig_file function.

    This test verifies that:
    1. The function correctly deletes the specified file asynchronously
    2. The function returns the file path
    3. The file no longer exists after deletion

    Args:
        tmp_path: Pytest fixture providing temporary directory path
    """
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    assert test_file.exists()

    result = await aunlink_orig_file(str(test_file))
    assert result == str(test_file)
    assert not test_file.exists()


@pytest.mark.asyncio
async def test_aunlink_orig_file_nonexistent(tmp_path: pathlib.Path) -> None:
    """Test aunlink_orig_file with a nonexistent file.

    This test verifies that the function raises FileNotFoundError
    when attempting to delete a nonexistent file asynchronously.

    Args:
        tmp_path: Pytest fixture providing temporary directory path
    """
    nonexistent_file = tmp_path / "nonexistent.txt"

    with pytest.raises(FileNotFoundError):
        await aunlink_orig_file(str(nonexistent_file))
