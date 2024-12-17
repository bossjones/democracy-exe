"""Tests for media compression utilities."""

from __future__ import annotations

import os
import pathlib

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

from loguru import logger

import pytest

from democracy_exe.shell import ProcessException
from democracy_exe.utils.twitter_utils.compression import (
    AUDIO_EXTENSIONS,
    MAX_DISCORD_SIZE,
    VIDEO_EXTENSIONS,
    CompressionError,
    compress_media,
    is_compressible,
)


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture()
def sample_video() -> pathlib.Path:
    """Get path to sample video file for testing.

    Returns:
        Path to sample video file
    """
    return pathlib.Path("tests/fixtures/song.mp4")


@pytest.fixture()
def sample_audio() -> pathlib.Path:
    """Get path to sample audio file for testing.

    Returns:
        Path to sample audio file
    """
    return pathlib.Path("tests/fixtures/song.mp3")


@pytest.fixture
def mock_file(tmp_path: pathlib.Path, sample_video: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    """Create a mock file for testing based on sample video.

    Args:
        tmp_path: Pytest temporary directory fixture
        sample_video: Sample video fixture

    Yields:
        Path to mock file
    """
    file_path = tmp_path / sample_video.name
    # Create a file with content to ensure non-zero size
    file_path.write_bytes(b"x" * 1024)  # 1KB file
    yield file_path


@pytest.fixture
def large_mock_file(tmp_path: pathlib.Path, sample_video: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    """Create a large mock file exceeding Discord size limit.

    Args:
        tmp_path: Pytest temporary directory fixture
        sample_video: Sample video fixture

    Yields:
        Path to large mock file
    """
    file_path = tmp_path / f"large_{sample_video.name}"
    # Create a file larger than MAX_DISCORD_SIZE
    file_path.write_bytes(b"x" * (MAX_DISCORD_SIZE + 1024))
    yield file_path


@pytest.fixture
def mock_script(tmp_path: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    """Create a mock compression script.

    Args:
        tmp_path: Pytest temporary directory fixture

    Yields:
        Path to mock script
    """
    script_path = tmp_path / "compress-discord.sh"
    script_content = """#!/bin/bash
    cp "$1" "$(dirname "$1")/25MB_$(basename "$1" | cut -f 1 -d '.').mp4"
    """
    script_path.write_text(script_content)
    script_path.chmod(0o755)
    yield script_path


@pytest.mark.asyncio
async def test_compress_media_small_file(
    mock_file: pathlib.Path,
    caplog: LogCaptureFixture,
) -> None:
    """Test compression of a file smaller than Discord limit.

    Args:
        mock_file: Mock file fixture
        caplog: Pytest log capture fixture
    """
    result = await compress_media(mock_file)
    assert result == str(mock_file)
    assert "Successfully compressed" not in caplog.text


@pytest.mark.asyncio
async def test_compress_media_large_file(
    large_mock_file: pathlib.Path,
    mock_script: pathlib.Path,
    caplog: LogCaptureFixture,
) -> None:
    """Test compression of a file larger than Discord limit.

    Args:
        large_mock_file: Large mock file fixture
        mock_script: Mock script fixture
        caplog: Pytest log capture fixture
    """
    result = await compress_media(large_mock_file, script_path=str(mock_script))
    expected_output = large_mock_file.parent / f"25MB_{large_mock_file.stem}.mp4"
    assert result == str(expected_output)
    # assert "Successfully compressed" in caplog.text


@pytest.mark.asyncio
async def test_compress_media_with_sample_video(
    sample_video: pathlib.Path,
    mock_script: pathlib.Path,
    caplog: LogCaptureFixture,
) -> None:
    """Test compression using actual sample video file.

    Args:
        sample_video: Sample video fixture
        mock_script: Mock script fixture
        caplog: Pytest log capture fixture
    """
    if not sample_video.exists():
        pytest.skip("Sample video file not found")

    result = await compress_media(sample_video, script_path=str(mock_script))
    if sample_video.stat().st_size > MAX_DISCORD_SIZE:
        expected_output = sample_video.parent / f"25MB_{sample_video.stem}.mp4"
        assert result == str(expected_output)
        # assert "Successfully compressed" in caplog.text
    else:
        assert result == str(sample_video)
        # assert "Successfully compressed" not in caplog.text


@pytest.mark.asyncio
async def test_compress_media_nonexistent_file() -> None:
    """Test compression of a nonexistent file."""
    with pytest.raises(CompressionError, match="File not found"):
        await compress_media("nonexistent.mp4")


@pytest.mark.asyncio
async def test_compress_media_missing_script(
    large_mock_file: pathlib.Path,
) -> None:
    """Test compression with missing script.

    Args:
        large_mock_file: Large mock file fixture
    """
    with pytest.raises(CompressionError, match="Compression script not found"):
        await compress_media(large_mock_file, script_path="nonexistent.sh")


@pytest.mark.asyncio
async def test_compress_media_process_error(
    large_mock_file: pathlib.Path,
    mock_script: pathlib.Path,
    mocker: MockerFixture,
) -> None:
    """Test compression with process error.

    Args:
        large_mock_file: Large mock file fixture
        mock_script: Mock script fixture
        mocker: Pytest mocker fixture
    """
    # Mock run_coroutine_subprocess to raise ProcessException
    mocker.patch(
        "democracy_exe.utils.twitter_utils.compression.run_coroutine_subprocess",
        side_effect=ProcessException("Mock error"),
    )

    with pytest.raises(CompressionError, match="Failed to compress"):
        await compress_media(large_mock_file, script_path=str(mock_script))


@pytest.mark.parametrize(
    "file_path,expected",
    [
        ("video.mp4", True),
        ("audio.mp3", True),
        ("image.jpg", False),
        ("document.pdf", False),
        ("VIDEO.MP4", True),
        ("AUDIO.MP3", True),
    ],
)
def test_is_compressible(file_path: str, expected: bool) -> None:
    """Test file compressibility check.

    Args:
        file_path: Path to test
        expected: Expected result
    """
    assert is_compressible(file_path) == expected


def test_video_extensions() -> None:
    """Test video extensions set."""
    assert ".mp4" in VIDEO_EXTENSIONS
    assert ".avi" in VIDEO_EXTENSIONS
    assert ".mkv" in VIDEO_EXTENSIONS


def test_audio_extensions() -> None:
    """Test audio extensions set."""
    assert ".mp3" in AUDIO_EXTENSIONS
    assert ".wav" in AUDIO_EXTENSIONS
    assert ".m4a" in AUDIO_EXTENSIONS
