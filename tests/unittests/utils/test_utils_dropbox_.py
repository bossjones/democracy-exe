# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
# pylint: disable=no-member
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest_asyncio
import requests

from dropbox.exceptions import ApiError, AuthError

import pytest

from pytest_mock import MockerFixture

from democracy_exe import aio_settings
from democracy_exe.utils import dropbox_
from democracy_exe.utils.dropbox_ import (
    BadInputException,
    cli_oauth,
    co_upload_to_dropbox,
    create_session,
    download_img,
    get_dropbox_client,
    iter_dir_and_upload,
    list_files_in_remote_folder,
    select_revision,
)
from democracy_exe.utils.file_functions import tilda


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch


APP_KEY = "dummy_app_key"
APP_SECRET = "dummy_app_secret"
ACCESS_TOKEN = "dummy_access_token"
REFRESH_TOKEN = "dummy_refresh_token"
EXPIRES_IN = 14400
ACCOUNT_ID = "dummy_account_id"
USER_ID = "dummy_user_id"
ADMIN_ID = "dummy_admin_id"
TEAM_MEMBER_ID = "dummy_team_member_id"
SCOPE_LIST = ["files.metadata.read", "files.metadata.write"]
EXPIRATION = datetime.now(UTC) + timedelta(seconds=EXPIRES_IN)

EXPIRATION_BUFFER = timedelta(minutes=5)

# pylint: disable=protected-access

is_running_in_github = bool(os.environ.get("GITHUB_ACTOR"))


# TODO: mock this correctly
@pytest.fixture
async def temp_file() -> AsyncGenerator[Path, None]:
    """Create a temporary test file.

    Yields:
        Path: Path to temporary file
    """
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(b"test content")
        path = Path(f.name)
    yield path
    path.unlink()


@pytest.mark.dropboxonly
@pytest.mark.unittest
class TestDropboxClient:
    """Test Dropbox client functionality."""

    @pytest.mark.asyncio
    async def test_list_files_in_remote_folder(self, mocker: MockerFixture) -> None:
        """Test listing files in remote folder."""
        mock_dbx = mocker.MagicMock()
        mock_entries = [mocker.MagicMock(name="file1.txt"), mocker.MagicMock(name="file2.txt")]
        mock_dbx.files_list_folder.return_value.entries = mock_entries

        await list_files_in_remote_folder(mock_dbx)

        mock_dbx.files_list_folder.assert_called_once()

    @pytest.mark.asyncio
    async def test_download_img(self, mocker: MockerFixture, temp_file: Path) -> None:
        """Test downloading image from Dropbox."""
        mock_dbx = mocker.MagicMock()
        mock_dbx.files_download.return_value = (mocker.MagicMock(), mocker.MagicMock(content=b"image data"))

        await download_img(mock_dbx, temp_file.name)

        mock_dbx.files_download.assert_called_once()

    @pytest.mark.asyncio
    async def test_iter_dir_and_upload(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """Test directory iteration and upload."""
        mock_dbx = mocker.MagicMock()
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        await iter_dir_and_upload(mock_dbx, "/remote", str(tmp_path))

        mock_dbx.files_upload.assert_called()

    @pytest.mark.asyncio
    async def test_co_upload_to_dropbox(self, mocker: MockerFixture, temp_file: Path) -> None:
        """Test cooperative upload to Dropbox."""
        mock_dbx = mocker.MagicMock()

        await co_upload_to_dropbox(mock_dbx, str(temp_file), "/remote")

        mock_dbx.files_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_select_revision(self, mocker: MockerFixture) -> None:
        """Test revision selection."""
        mock_dbx = mocker.MagicMock()
        result = select_revision(mock_dbx, "test.txt", "/folder")
        assert result == "/folder/test.txt"

    def test_cli_oauth_success(self, mocker: MockerFixture) -> None:
        """Test successful OAuth flow."""
        # Mock the OAuth flow
        mock_flow = mocker.MagicMock()
        mock_flow.start.return_value = "https://dropbox.com/oauth"
        mock_flow.finish.return_value = mocker.MagicMock(access_token="test_token")  # noqa: S106

        # Mock Dropbox client
        mock_dbx = mocker.MagicMock()
        mocker.patch("dropbox.Dropbox", return_value=mock_dbx)
        mocker.patch("dropbox.DropboxOAuth2FlowNoRedirect", return_value=mock_flow)
        mocker.patch("builtins.input", return_value="test_code")
        mocker.patch("builtins.print")  # Suppress output

        # Execute
        cli_oauth()

        # Verify
        mock_flow.start.assert_called_once()
        mock_flow.finish.assert_called_once_with("test_code")
        mock_dbx.users_get_current_account.assert_called_once()

    def test_cli_oauth_failure(self, mocker: MockerFixture) -> None:
        """Test OAuth flow failure."""
        # Mock the OAuth flow
        mock_flow = mocker.MagicMock()
        mock_flow.start.return_value = "https://dropbox.com/oauth"
        mock_flow.finish.side_effect = Exception("OAuth failed")

        mocker.patch("dropbox.DropboxOAuth2FlowNoRedirect", return_value=mock_flow)
        mocker.patch("builtins.input", return_value="test_code")
        mocker.patch("builtins.print")  # Suppress output

        # Mock sys.exit to avoid test termination
        mock_exit = mocker.patch("sys.exit")

        # Execute
        cli_oauth()

        # Verify
        mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_get_dropbox_client_auth_error(self, mocker: MockerFixture) -> None:
        """Test client initialization with auth error."""
        mock_dbx = mocker.MagicMock()
        mock_dbx.users_get_current_account.side_effect = AuthError(request_id="test_id", error="test_error")
        mocker.patch("dropbox.Dropbox", return_value=mock_dbx)

        with pytest.raises(SystemExit):
            await get_dropbox_client("invalid_token")

    @pytest.mark.asyncio
    async def test_get_dropbox_client_api_error(self, mocker: MockerFixture) -> None:
        """Test client initialization with API error."""
        mock_dbx = mocker.MagicMock()
        mock_dbx.users_get_current_account.side_effect = ApiError(
            request_id="test_id", error="test_error", user_message_locale="en", user_message_text="Test error message"
        )
        # Configure mock to return None
        mock_dbx_factory = mocker.patch("dropbox.Dropbox")
        mock_dbx_factory.return_value = mock_dbx

        result = await get_dropbox_client("test_token")
        assert result is None
        mock_dbx.users_get_current_account.assert_called_once()

    @pytest.fixture(scope="function")
    def session_instance(self, mocker: MockerFixture) -> requests.Session:
        session_obj = create_session()
        post_response = mocker.MagicMock(status_code=200)
        post_response.json.return_value = {
            "access_token": ACCESS_TOKEN,
            "expires_in": EXPIRES_IN,
        }
        mocker.patch.object(session_obj, "post", return_value=post_response)
        return session_obj

    @pytest.fixture(scope="function")
    def invalid_grant_session_instance(self, mocker: MockerFixture) -> requests.Session:
        session_obj = create_session()
        post_response = mocker.MagicMock(status_code=400)
        post_response.json.return_value = {"error": "invalid_grant"}
        mocker.patch.object(session_obj, "post", return_value=post_response)
        return session_obj

    @pytest_asyncio.fixture
    async def test_default_dropbox_raises_assertion_error(self, monkeypatch: MonkeyPatch) -> None:
        # import bpdb
        # bpdb.set_trace()
        # paranoid about weird libraries trying to read env vars during testing
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_DROPBOX_CEREBRO_TOKEN", "fake_dropbox_token")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_DROPBOX_CEREBRO_APP_KEY", "fake_dropbox_app_key")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_DROPBOX_CEREBRO_APP_SECRET", "fake_dropbox_app_secret")
        monkeypatch.setenv("DEMOCRACY_EXE_CONFIG_DROPBOX_CEREBRO_OAUTH_ACCESS_TOKEN", "fake_dropbox_oauth_access_token")
        await asyncio.sleep(0.05)

        test_settings: aio_settings.AioSettings = aio_settings.AioSettings()
        assert test_settings.dropbox_cerebro_token == 1337
        assert test_settings.dropbox_cerebro_app_key == 1337
        assert test_settings.dropbox_cerebro_app_secret == 1337
        assert test_settings.dropbox_cerebro_token == 1337

        with pytest.raises(BadInputException):
            # Requires either access token or refresh token
            await dropbox_.get_dropbox_client()
