from __future__ import annotations

import os
import zipfile

from typing import TYPE_CHECKING

import pytest

from democracy_exe.utils.ai_docs_utils.extract_repo import make_zip_file


if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.fixtures import FixtureRequest
    from _pytest.tmpdir import TempPathFactory


@pytest.fixture
def test_directory(tmp_path: Path) -> Path:
    """Create a test directory with some files.

    Args:
        tmp_path: Pytest fixture providing temporary directory

    Returns:
        Path: Path to test directory with sample files
    """
    # Create test directory structure
    test_dir = tmp_path / "test_repo"
    test_dir.mkdir()

    # Create some test files
    (test_dir / "file1.txt").write_text("Test content 1")
    (test_dir / "file2.txt").write_text("Test content 2")

    return test_dir


def test_make_zip_file(test_directory: Path, tmp_path: Path) -> None:
    """Test make_zip_file function creates zip archive correctly.

    Args:
        test_directory: Fixture providing directory with test files
        tmp_path: Pytest fixture providing temporary directory
    """
    # Setup test parameters
    output_name = str(tmp_path / "test_output")

    # Call function under test
    zip_path = make_zip_file(str(test_directory), output_name)

    # Verify zip file was created with correct name
    assert zip_path == f"{output_name}.zip"
    assert os.path.exists(zip_path)

    # Verify zip contents
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # Get list of files in zip
        zip_files = zip_ref.namelist()

        # Verify expected files exist in zip
        assert "file1.txt" in zip_files
        assert "file2.txt" in zip_files

        # Verify file contents
        assert zip_ref.read("file1.txt").decode() == "Test content 1"
        assert zip_ref.read("file2.txt").decode() == "Test content 2"
