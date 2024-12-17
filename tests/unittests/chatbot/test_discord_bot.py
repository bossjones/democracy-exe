from __future__ import annotations

import os
import pathlib
import re

from typing import TYPE_CHECKING

import pytest

from democracy_exe.chatbot.discord_bot import unlink_orig_file


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


def test_unlink_orig_file(
    mock_file: pathlib.Path,
    capsys: CaptureFixture,
    mocker: MockerFixture,
) -> None:
    """Test the unlink_orig_file function.

    This test verifies that:
    1. The function correctly deletes the specified file
    2. The function returns the file path
    3. The function prints the expected deletion message

    Args:
        mock_file: Fixture providing path to temporary test file
        capsys: Pytest fixture for capturing stdout/stderr
        mocker: Pytest fixture for mocking
    """
    # Verify file exists before deletion
    assert mock_file.exists()

    # Call the function
    result = unlink_orig_file(str(mock_file))

    # Verify the function returned the correct path
    assert result == str(mock_file)

    # Verify file was deleted
    assert not mock_file.exists()

    # Verify correct message was printed
    captured = capsys.readouterr()
    expected_output = f"deleting ... {mock_file}\n"
    assert captured.out == expected_output


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
