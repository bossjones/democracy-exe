from __future__ import annotations

import datetime
import os
import textwrap
import zipfile

from typing import TYPE_CHECKING

import pytest

from democracy_exe.utils.ai_docs_utils.extract_repo import (
    extract_local_directory,
    has_sufficient_content,
    is_desired_file,
    is_likely_useful_file,
    make_zip_file,
    remove_comments_and_docstrings,
)


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


@pytest.mark.parametrize(
    "file_content,min_line_count,expected,reason",
    [
        # Sufficient content (using default min_line_count=5)
        (
            [
                "def example():",
                "    x = 1",
                "    y = 2",
                "    z = 3",
                "    return x + y + z",
            ],
            5,
            True,
            "Exactly minimum required lines",
        ),
        (
            [
                "import os",
                "import sys",
                "def main():",
                "    print('hello')",
                "    return 0",
                "if __name__ == '__main__':",
                "    main()",
            ],
            5,
            True,
            "More than minimum lines",
        ),
        # Insufficient content
        (
            ["def test():", "    pass", "    return None"],
            5,
            False,
            "Less than minimum lines",
        ),
        # Mixed content with comments and empty lines
        (
            [
                "// Header comment",
                "",
                "def example():",
                "    // Internal comment",
                "    x = 1",
                "",
                "    y = 2",
                "    return x + y",
            ],
            5,
            False,
            "Not enough non-comment lines",
        ),
        (
            [
                "// Comment 1",
                "line 1",
                "",
                "// Comment 2",
                "line 2",
                "line 3",
                "// Comment 3",
                "line 4",
                "line 5",
                "",
            ],
            5,
            True,
            "Sufficient content with mixed comments and empty lines",
        ),
        # Custom min_line_count
        (
            ["def test():", "    return True"],
            2,
            True,
            "Custom minimum (2 lines) met",
        ),
        (
            ["single_line = True"],
            1,
            True,
            "Custom minimum (1 line) met",
        ),
        # Edge cases
        ([], 5, False, "Empty file"),
        (["", "", "", ""], 5, False, "Only empty lines"),
        (["// Comment 1", "// Comment 2", "// Comment 3"], 5, False, "Only comments"),
        (
            ["    ", "\t", "\n", "  \t  \n  "],
            5,
            False,
            "Only whitespace",
        ),
    ],
)
def test_has_sufficient_content(
    file_content: list[str],
    min_line_count: int,
    expected: bool,
    reason: str,
) -> None:
    """Test has_sufficient_content function correctly identifies files with sufficient content.

    Args:
        file_content: List of lines to check
        min_line_count: Minimum number of substantive lines required
        expected: Expected result
        reason: Description of what the test case is checking
    """
    result = has_sufficient_content(file_content, min_line_count)
    assert result == expected, (
        f"Failed case: {reason} - Expected has_sufficient_content() to be {expected}, got {result}"
        f"\nContent:\n{chr(10).join(file_content)}"
    )


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
@pytest.mark.parametrize(
    "source,expected,reason",
    [
        # Function docstrings
        (
            textwrap.dedent('''
                def example():
                    """This is a function docstring."""
                    return True
            ''').strip(),
            textwrap.dedent("""
                def example():
                    return True
            """).strip(),
            "Simple function docstring",
        ),
        # Class docstrings
        (
            textwrap.dedent('''
                class Example:
                    """This is a class docstring."""
                    def __init__(self):
                        """Constructor docstring."""
                        pass
            ''').strip(),
            textwrap.dedent("""
                class Example:
                    def __init__(self):
                        pass
            """).strip(),
            "Class and method docstrings",
        ),
        # Module docstrings
        (
            textwrap.dedent('''
                """This is a module docstring."""

                import os

                def main():
                    pass
            ''').strip(),
            textwrap.dedent("""
                import os

                def main():
                    pass
            """).strip(),
            "Module level docstring",
        ),
        # Async function docstrings
        (
            textwrap.dedent('''
                async def fetch():
                    """This is an async function docstring."""
                    return await something()
            ''').strip(),
            textwrap.dedent("""
                async def fetch():
                    return await something()
            """).strip(),
            "Async function docstring",
        ),
        # Mixed content
        (
            textwrap.dedent('''
                """Module docstring."""

                class Example:
                    """Class docstring."""

                    def method(self):
                        """Method docstring."""
                        # This is a comment
                        x = 1  # Inline comment
                        return x
            ''').strip(),
            textwrap.dedent("""
                class Example:
                    def method(self):
                        x = 1
                        return x
            """).strip(),
            "Mixed docstrings and comments",
        ),
        # Complex nested structures
        (
            textwrap.dedent('''
                class Outer:
                    """Outer class docstring."""

                    class Inner:
                        """Inner class docstring."""

                        def inner_method(self):
                            """Inner method docstring."""
                            return True

                    def outer_method(self):
                        """Outer method docstring."""
                        return False
            ''').strip(),
            textwrap.dedent("""
                class Outer:
                    class Inner:
                        def inner_method(self):
                            return True

                    def outer_method(self):
                        return False
            """).strip(),
            "Nested class and method docstrings",
        ),
        # Edge cases
        (
            "x = 1  # Simple assignment",
            "x = 1",
            "Single line with comment",
        ),
        (
            textwrap.dedent("""
                def no_docstring():
                    return True
            """).strip(),
            textwrap.dedent("""
                def no_docstring():
                    return True
            """).strip(),
            "Function without docstring",
        ),
        (
            textwrap.dedent("""
                class NoDocstring:
                    def __init__(self):
                        pass
            """).strip(),
            textwrap.dedent("""
                class NoDocstring:
                    def __init__(self):
                        pass
            """).strip(),
            "Class without docstring",
        ),
    ],
)
def test_remove_comments_and_docstrings(source: str, expected: str, reason: str) -> None:
    """Test remove_comments_and_docstrings function correctly removes docstrings and comments.

    Args:
        source: Source code to process
        expected: Expected result after removing comments and docstrings
        reason: Description of what the test case is checking
    """
    result = remove_comments_and_docstrings(source)
    # Normalize whitespace for comparison
    result = textwrap.dedent(result).strip()
    expected = textwrap.dedent(expected).strip()

    assert result == expected, f"Failed case: {reason}\nExpected:\n{expected}\nGot:\n{result}"


@pytest.fixture
def test_repo_directory(tmp_path: Path) -> Path:
    """Create a test repository directory with various types of files.

    Args:
        tmp_path: Pytest fixture providing temporary directory

    Returns:
        Path: Path to test repository directory
    """
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()

    # Create source directory with valid files
    src_dir = repo_dir / "src"
    src_dir.mkdir()

    # Valid Python file with sufficient content
    valid_py = src_dir / "main.py"
    valid_py.write_text(
        textwrap.dedent('''
        def main():
            """Main function."""
            print("Hello")
            x = 1
            y = 2
            return x + y

        if __name__ == "__main__":
            main()
    ''')
    )

    # Valid TypeScript file
    valid_ts = src_dir / "component.ts"
    valid_ts.write_text(
        textwrap.dedent("""
        export class Component {
            private value: number;

            constructor() {
                this.value = 42;
            }

            getValue(): number {
                return this.value;
            }
        }
    """)
    )

    # Create test directory (should be excluded)
    test_dir = repo_dir / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_main.py"
    test_file.write_text("def test_main(): pass")

    # Create docs directory (should be excluded)
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir()
    docs_file = docs_dir / "api.md"
    docs_file.write_text("# API Documentation")

    # Create hidden directory (should be excluded)
    hidden_dir = repo_dir / ".hidden"
    hidden_dir.mkdir()
    hidden_file = hidden_dir / "config.py"
    hidden_file.write_text("SECRET = 'secret'")

    # Create file with insufficient content (should be excluded)
    small_file = src_dir / "small.py"
    small_file.write_text("x = 1\n")

    # Create package.json (should be included)
    pkg_json = repo_dir / "package.json"
    pkg_json.write_text('{"name": "test-repo", "version": "1.0.0"}')

    return repo_dir


def test_extract_local_directory(test_repo_directory: Path, tmp_path: Path) -> None:
    """Test extract_local_directory function processes files correctly.

    Args:
        test_repo_directory: Fixture providing test repository
        tmp_path: Pytest fixture providing temporary directory
    """
    # Get the original working directory
    original_cwd = os.getcwd()

    try:
        # Change to test directory to ensure relative paths work
        os.chdir(str(test_repo_directory))

        # Call function under test with relative path
        output_file = extract_local_directory(".")

        # Verify output file name
        assert output_file == "test_repo_code.txt"
        assert os.path.exists(output_file)

        # Read output file content
        with open(output_file, encoding="utf-8") as f:
            content = f.read()
            print(f"\nOutput file contents:\n{content}")  # Debug output

        # Verify included files (using normalized paths)
        assert "# File: src/main.py" in content or "# File: ./src/main.py" in content
        assert "def main():" in content
        assert "# File: src/component.ts" in content or "# File: ./src/component.ts" in content
        assert "export class Component" in content
        assert "# File: package.json" in content or "# File: ./package.json" in content
        assert '"name": "test-repo"' in content

        # Verify excluded files
        assert "test_main.py" not in content
        assert "api.md" not in content
        assert "config.py" not in content
        assert "small.py" not in content

    finally:
        # Restore original working directory
        os.chdir(original_cwd)
