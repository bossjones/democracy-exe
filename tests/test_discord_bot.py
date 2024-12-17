"""Tests for discord_bot.py functionality."""

from __future__ import annotations

import os
import pathlib

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, List

from loguru import logger

import pytest

from democracy_exe.chatbot.utils.discord_utils import aio_extensions


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
    # Create mock cogs directory structure
    cogs_dir = tmp_path / "democracy_exe" / "chatbot" / "cogs"
    cogs_dir.mkdir(parents=True)

    # Create some test cog files
    (cogs_dir / "test_cog1.py").touch()
    (cogs_dir / "test_cog2.py").touch()
    (cogs_dir / "not_a_cog.txt").touch()

    # Create a subdirectory with more cogs
    sub_dir = cogs_dir / "subcategory"
    sub_dir.mkdir()
    (sub_dir / "test_cog3.py").touch()

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
    monkeypatch.setattr("democracy_exe.chatbot.discord_bot.HERE", str(mock_cogs_directory.parent))

    # Collect all yielded extensions
    extensions = []
    async for ext in aio_extensions():
        extensions.append(ext)

    # Verify expected module paths are found
    expected = [
        "democracy_exe.chatbot.cogs.test_cog1",
        "democracy_exe.chatbot.cogs.test_cog2",
        "democracy_exe.chatbot.cogs.subcategory.test_cog3",
    ]

    assert sorted(extensions) == sorted(expected)

    # # Verify logging
    # assert "Starting async extension discovery" in caplog.text
    # assert "Successfully initialized async file search" in caplog.text
    # assert "Completed async extension discovery" in caplog.text


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
    monkeypatch.setattr("democracy_exe.chatbot.discord_bot.HERE", str(tmp_path))

    with pytest.raises(Exception) as exc_info:
        async for _ in aio_extensions():
            pass

    assert "Extension discovery failed" in str(exc_info.value)
    assert "Error discovering extensions" in caplog.text


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
    monkeypatch.setattr("democracy_exe.chatbot.discord_bot.HERE", str(mock_cogs_directory.parent))

    # Make one file unreadable
    unreadable_file = mock_cogs_directory / "test_cog1.py"
    os.chmod(unreadable_file, 0o000)

    # Should skip unreadable file but continue processing others
    extensions = []
    async for ext in aio_extensions():
        extensions.append(ext)

    assert "test_cog1" not in " ".join(extensions)
    assert "Skipping inaccessible extension file" in caplog.text

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
    cogs_dir = tmp_path / "democracy_exe" / "chatbot" / "cogs"
    cogs_dir.mkdir(parents=True)

    monkeypatch.setattr("democracy_exe.chatbot.discord_bot.HERE", str(cogs_dir.parent))

    # Should yield no extensions
    extensions = []
    async for ext in aio_extensions():
        extensions.append(ext)

    assert len(extensions) == 0
    assert "Successfully initialized async file search" in caplog.text
    assert "Completed async extension discovery" in caplog.text
