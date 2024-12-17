"""Tests for Discord bot functionality.

This module contains tests for the Discord bot's core functionality.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

from typing import TYPE_CHECKING, Any, List

import pytest

from democracy_exe.chatbot.utils.extension_utils import aio_extensions


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_cogs_directory(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a mock cogs directory with test files.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to mock cogs directory
    """
    cogs_dir = tmp_path / "chatbot" / "cogs"
    cogs_dir.mkdir(parents=True)

    # Create test cog files
    (cogs_dir / "test_cog1.py").write_text("# Test cog 1")
    (cogs_dir / "test_cog2.py").write_text("# Test cog 2")
    (cogs_dir / "__init__.py").write_text("")

    # Create subdirectory with another cog
    subcategory = cogs_dir / "subcategory"
    subcategory.mkdir()
    (subcategory / "test_cog3.py").write_text("# Test cog 3")
    (subcategory / "__init__.py").write_text("")

    return cogs_dir


@pytest.mark.asyncio
async def test_aio_extensions_finds_cogs(
    mock_cogs_directory: pathlib.Path, mocker: MockerFixture, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Test that aio_extensions finds and yields correct module paths.

    Args:
        mock_cogs_directory: Fixture providing mock cogs directory
        mocker: Pytest mocker fixture
        monkeypatch: Pytest monkeypatch fixture
        caplog: Pytest log capture fixture
    """
    # Mock HERE to point to our test directory
    monkeypatch.setattr("democracy_exe.chatbot.utils.extension_utils.HERE", str(mock_cogs_directory.parent))

    # Collect all yielded extensions
    extensions = []
    async for ext in aio_extensions():
        extensions.append(ext)

    # Verify expected module paths are found
    expected = [
        "cogs.test_cog1",
        "cogs.test_cog2",
        "cogs.subcategory.test_cog3",
    ]

    assert sorted(extensions) == sorted(expected)
    # assert "Successfully initialized async file search" in caplog.text


@pytest.mark.asyncio
async def test_aio_extensions_handles_missing_directory(
    tmp_path: pathlib.Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Test that aio_extensions handles missing cogs directory gracefully.

    Args:
        tmp_path: Pytest temporary directory fixture
        monkeypatch: Pytest monkeypatch fixture
        caplog: Pytest log capture fixture
    """
    # Point to non-existent directory
    monkeypatch.setattr("democracy_exe.chatbot.utils.extension_utils.HERE", str(tmp_path))

    with pytest.raises(FileNotFoundError) as exc_info:
        async for _ in aio_extensions():
            pass

    assert "Cogs directory not found" in str(exc_info.value)
    # assert "Error discovering extensions" in caplog.text


@pytest.mark.asyncio
async def test_aio_extensions_handles_unreadable_file(
    mock_cogs_directory: pathlib.Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Test that aio_extensions handles unreadable files appropriately.

    Args:
        mock_cogs_directory: Fixture providing mock cogs directory
        monkeypatch: Pytest monkeypatch fixture
        caplog: Pytest log capture fixture
    """
    monkeypatch.setattr("democracy_exe.chatbot.utils.extension_utils.HERE", str(mock_cogs_directory.parent))

    # Make one file unreadable
    unreadable_file = mock_cogs_directory / "test_cog1.py"
    os.chmod(unreadable_file, 0o000)

    # Should skip unreadable file but continue processing others
    extensions = []
    async for ext in aio_extensions():
        extensions.append(ext)

    assert "test_cog1" not in " ".join(extensions)
    # assert "Skipping inaccessible extension file" in caplog.text

    # Cleanup
    os.chmod(unreadable_file, 0o644)


@pytest.mark.asyncio
async def test_aio_extensions_empty_directory(
    tmp_path: pathlib.Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """Test that aio_extensions handles empty cogs directory appropriately.

    Args:
        tmp_path: Pytest temporary directory fixture
        monkeypatch: Pytest monkeypatch fixture
        caplog: Pytest log capture fixture
    """
    # Create empty cogs directory
    cogs_dir = tmp_path / "chatbot" / "cogs"
    cogs_dir.mkdir(parents=True)
    monkeypatch.setattr("democracy_exe.chatbot.utils.extension_utils.HERE", str(tmp_path))

    with pytest.raises(FileNotFoundError):
        # Should yield no extensions
        extensions = []
        async for ext in aio_extensions():
            extensions.append(ext)

        assert len(extensions) == 0
        # assert "Successfully initialized async file search" in caplog.text
