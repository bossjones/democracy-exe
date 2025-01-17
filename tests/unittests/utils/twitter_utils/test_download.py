"""Tests for twitter download utilities."""

from __future__ import annotations

import json
import logging
import pathlib

from typing import TYPE_CHECKING, Any, Dict

import structlog


logger = structlog.get_logger(__name__)

import pytest

from democracy_exe.utils.twitter_utils.download import _parse_tweet_metadata, download_tweet
from democracy_exe.utils.twitter_utils.types import DownloadResult, TweetDownloadMode


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_info_json(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a mock info.json file.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to mock info.json file
    """
    info_data = {
        "tweet_id": 123456789,
        "conversation_id": 123456789,
        "date": "2024-03-10T12:00:00Z",
        "author": {
            "id": 251084543,
            "name": "Test User",
            "nick": "Test Nick",
            "date": "2011-02-12 12:07:13",
            "profile_image": "https://example.com/profile.jpg",
            "favourites_count": 21833,
            "followers_count": 73026,
            "friends_count": 10009,
            "listed_count": 66,
            "media_count": 17013,
            "statuses_count": 423991,
            "description": "Test description",
        },
        "user": {
            "id": 251084543,
            "name": "Test User",
            "nick": "Test Nick",
            "date": "2011-02-12 12:07:13",
            "profile_image": "https://example.com/profile.jpg",
            "favourites_count": 21833,
            "followers_count": 73026,
            "friends_count": 10009,
            "listed_count": 66,
            "media_count": 17013,
            "statuses_count": 423991,
            "description": "Test description",
        },
        "lang": "en",
        "source": "Twitter Web App",
        "content": "Test tweet content",
        "subcategory": "tweet",
    }

    # Create info.json
    info_path = tmp_path / "info.json"
    info_path.write_text(json.dumps(info_data))

    # Create a test image file
    test_image = tmp_path / "test.jpg"
    test_image.touch()

    return info_path


@pytest.mark.asyncio
class TestDownloadTweet:
    """Test suite for tweet download functionality."""

    async def test_download_tweet_success(
        self, mocker: MockerFixture, tmp_path: pathlib.Path, mock_info_json: pathlib.Path
    ) -> None:
        """Test successful tweet download.

        Args:
            mocker: Pytest mocker fixture
            tmp_path: Pytest temporary directory fixture
            mock_info_json: Mock info.json fixture
        """
        # Mock shell command execution with AsyncMock
        mock_shell = mocker.AsyncMock(return_value=(b"", b""))
        mocker.patch("democracy_exe.shell._aio_run_process_and_communicate", side_effect=mock_shell)

        # Get the test image created by mock_info_json fixture
        test_image = mock_info_json.parent / "test.jpg"

        # These are sync functions, so regular Mock is fine
        mocker.patch("democracy_exe.utils.file_functions.tree", return_value=[test_image])
        mocker.patch("democracy_exe.utils.file_functions.filter_media", return_value=[str(test_image)])

        # Test download
        result = await download_tweet(
            "https://twitter.com/user/status/123456789", mode="single", working_dir=str(tmp_path)
        )

        # Verify AsyncMock was called
        mock_shell.assert_awaited_once()

        # Verify result structure
        assert result["success"] is True
        assert result["metadata"]["id"] == "123456789"
        assert result["metadata"]["author"] == "Test User"
        assert result["metadata"]["content"] == "Test tweet content"
        assert len(result["local_files"]) == 1
        assert result["error"] is None

    async def test_download_tweet_failure(self, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
        """Test tweet download failure handling.

        Args:
            mocker: Pytest mocker fixture
            caplog: Pytest log capture fixture
        """
        with caplog.at_level(logging.DEBUG):
            # Mock shell command to raise error using AsyncMock
            mock_shell = mocker.AsyncMock(side_effect=Exception("Download failed"))
            mocker.patch("democracy_exe.shell._aio_run_process_and_communicate", side_effect=mock_shell)

            # Test download with error
            result: DownloadResult = await download_tweet("https://twitter.com/user/status/123456789", mode="single")

            # Verify AsyncMock was called
            mock_shell.assert_awaited_once()

            assert result["success"] is False
            assert result["error"] == "Download failed"
            assert not result["local_files"]
            # assert "Error downloading tweet" in caplog.text

    @pytest.mark.parametrize("mode", ["single", "thread", "card"])
    async def test_download_modes(self, mode: TweetDownloadMode, mocker: MockerFixture) -> None:
        """Test different download modes.

        Args:
            mode: Download mode to test
            mocker: Pytest mocker fixture
        """
        # Use AsyncMock for shell command
        mock_run = mocker.AsyncMock(return_value=(b"", b""))
        mocker.patch("democracy_exe.shell._aio_run_process_and_communicate", side_effect=mock_run)

        await download_tweet("https://twitter.com/user/status/123456789", mode=mode)

        # Verify AsyncMock was called
        mock_run.assert_awaited_once()

        # Verify correct command was used
        cmd_call = mock_run.call_args[0][0]
        if mode == "single":
            assert "gallery-dl" in " ".join(cmd_call)
        elif mode == "thread":
            assert "thread" in " ".join(cmd_call)
        elif mode == "card":
            assert "card" in " ".join(cmd_call)


class TestParseMetadata:
    """Test suite for metadata parsing functionality."""

    def test_parse_metadata_success(self, mock_info_json: pathlib.Path) -> None:
        """Test successful metadata parsing.

        Args:
            mock_info_json: Mock info.json fixture
        """
        metadata = _parse_tweet_metadata(str(mock_info_json.parent))

        assert metadata["id"] == "123456789"
        assert metadata["author"] == "Test User"
        assert metadata["content"] == "Test tweet content"
        assert len(metadata["media_urls"]) == 1
        assert metadata["created_at"] == "2024-03-10T12:00:00Z"

    def test_parse_metadata_missing_file(self, tmp_path: pathlib.Path, caplog: LogCaptureFixture) -> None:
        """Test metadata parsing with missing info file.

        Args:
            tmp_path: Pytest temporary directory fixture
            caplog: Pytest log capture fixture
        """
        metadata = _parse_tweet_metadata(str(tmp_path))

        assert metadata["id"] == ""
        assert metadata["author"] == ""
        assert not metadata["media_urls"]
        # assert "No info.json found" in caplog.text

    def test_parse_metadata_invalid_json(self, tmp_path: pathlib.Path, caplog: LogCaptureFixture) -> None:
        """Test metadata parsing with invalid JSON.

        Args:
            tmp_path: Pytest temporary directory fixture
            caplog: Pytest log capture fixture
        """
        # Create invalid JSON file
        info_path = tmp_path / "info.json"
        info_path.write_text("invalid json")

        # with pytest.raises((json.decoder.JSONDecodeError, ValueError,FileNotFoundError, KeyError), match="Expecting value") as exc_info:
        res = _parse_tweet_metadata(str(tmp_path))

        assert res["id"] == ""
        assert res["author"] == ""
        assert not res["media_urls"]
