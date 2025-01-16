from __future__ import annotations

import os
import zipfile

from typing import TYPE_CHECKING

import pytest

from democracy_exe.utils.ai_docs_utils.extract_repo import is_desired_file, is_likely_useful_file, make_zip_file


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


@pytest.mark.parametrize(
    "file_path,expected",
    [
        # Test file extensions
        ("src/file.ts", True),
        ("app/component.tsx", True),
        ("schema.prisma", True),
        ("script.py", True),
        # Test specific file names
        ("package.json", True),
        ("pyproject.toml", True),
        # Test negative cases
        ("file.txt", False),
        ("script.js", False),
        ("readme.md", False),
        ("requirements.txt", False),
        # Test with different path formats
        (os.path.join("src", "utils", "file.ts"), True),
        (os.path.join("src", "components", "file.jsx"), False),
        # Test case sensitivity
        ("FILE.PY", True),
        ("PACKAGE.JSON", True),
        ("file.TS", True),
    ],
)
def test_is_desired_file(file_path: str, expected: bool) -> None:
    """Test is_desired_file function correctly identifies desired files.

    Args:
        file_path: Path to test file
        expected: Expected result for the given file path
    """
    result = is_desired_file(file_path)
    assert result == expected, f"Expected is_desired_file('{file_path}') to be {expected}, got {result}"


@pytest.mark.parametrize(
    "file_path,expected,reason",
    [
        # Hidden directories/files (should be excluded)
        (".git/file.py", False, "Hidden directory"),
        ("src/.hidden/file.py", False, "Hidden directory"),
        (".env", False, "Hidden file"),
        # Test-related paths (should be excluded)
        ("tests/test_file.py", False, "Test directory"),
        ("src/tests/file.py", False, "Test directory"),
        ("src/test_utils.py", False, "Test file"),
        ("src/utils/test_helpers.py", False, "Test file"),
        # Excluded directories
        ("docs/api.md", False, "Docs directory"),
        ("examples/demo.py", False, "Examples directory"),
        ("__pycache__/module.pyc", False, "Pycache directory"),
        ("scripts/build.sh", False, "Scripts directory"),
        ("benchmarks/perf.py", False, "Benchmarks directory"),
        ("node_modules/package/index.js", False, "Node modules directory"),
        (".venv/lib/python3.8/site-packages/pkg.py", False, "Venv directory"),
        # Utility/config files
        ("src/hubconf.py", False, "Utility file"),
        ("setup.py", False, "Config file"),
        ("package-lock.json", False, "Lock file"),
        # GitHub workflow files
        ("workflows/stale.py", False, "GitHub workflow file"),
        ("docs/gen-card-model.py", False, "Card generation file"),
        ("scripts/write_model_card.py", False, "Model card file"),
        # Valid paths (should be included)
        ("src/main.py", True, "Source file"),
        ("lib/utils.py", True, "Library file"),
        ("app/components/Button.tsx", True, "Component file"),
        ("src/nested/deep/file.py", True, "Nested source file"),
        # Path separator handling
        (os.path.join("src", "utils", "file.py"), True, "OS-specific path separator"),
        (os.path.join("tests", "file.py"), False, "OS-specific excluded path"),
        # Edge cases
        ("test.py", False, "File with test in name"),
        ("contest.py", True, "File with test as substring"),
        ("src/testing.txt", False, "Non-code file with test in name"),
        ("src/latest.py", True, "File with test as substring"),
    ],
)
def test_is_likely_useful_file(file_path: str, expected: bool, reason: str) -> None:
    """Test is_likely_useful_file function correctly identifies useful files.

    Args:
        file_path: Path to test file
        expected: Expected result for the given file path
        reason: Description of what the test case is checking
    """
    result = is_likely_useful_file(file_path)
    assert result == expected, (
        f"Failed case: {reason} - Expected is_likely_useful_file('{file_path}') to be {expected}, got {result}"
    )
