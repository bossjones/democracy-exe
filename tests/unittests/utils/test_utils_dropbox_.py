# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
# pylint: disable=no-member
from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, cast
from urllib.parse import parse_qs, urlencode

import pytest_asyncio
import requests

from dropbox.exceptions import ApiError, AuthError
from langsmith import tracing_context
from loguru import logger

import pytest

from pytest_mock import MockerFixture

from democracy_exe import aio_settings
from democracy_exe.utils import dropbox_
from democracy_exe.utils._testing import ContextLogger
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
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from vcr.request import Request as VCRRequest


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


def _filter_request_headers(request: VCRRequest) -> Any:
    request.headers = {}
    # request.body =
    return request


def filter_sensitive_data(request: VCRRequest):
    request.headers = {}

    if request.body:
        # Parse the request body
        parsed_body = parse_qs(request.body)

        # Replace sensitive data
        if "client_id" in parsed_body:
            parsed_body["client_id"] = ["FILTERED"]
        if "client_secret" in parsed_body:
            parsed_body["client_secret"] = ["FILTERED"]

        # Encode the body back to a string
        request.body = urlencode(parsed_body, doseq=True)

    return request


def _filter_response(response: VCRRequest) -> VCRRequest:
    """
    Filter the response before recording.

    If the response has a 'retry-after' header, we set it to 0 to avoid waiting for the retry time.

    Args:
        response (VCRRequest): The response to filter.

    Returns:
        VCRRequest: The filtered response.
    """

    if "retry-after" in response["headers"]:
        response["headers"]["retry-after"] = "0"  # type: ignore
    if "x-stainless-arch" in response["headers"]:
        response["headers"]["x-stainless-arch"] = "arm64"  # type: ignore

    if "apim-request-id" in response["headers"]:
        response["headers"]["apim-request-id"] = ["9a705e27-2f04-4bd6-abd8-01848165ebbf"]  # type: ignore

    if "azureml-model-session" in response["headers"]:
        response["headers"]["azureml-model-session"] = ["d089-20240815073451"]  # type: ignore

    if "x-ms-client-request-id" in response["headers"]:
        response["headers"]["x-ms-client-request-id"] = ["9a705e27-2f04-4bd6-abd8-01848165ebbf"]  # type: ignore

    if "x-ratelimit-remaining-requests" in response["headers"]:
        response["headers"]["x-ratelimit-remaining-requests"] = ["144"]  # type: ignore
    if "x-ratelimit-remaining-tokens" in response["headers"]:
        response["headers"]["x-ratelimit-remaining-tokens"] = ["143324"]  # type: ignore
    if "x-request-id" in response["headers"]:
        response["headers"]["x-request-id"] = ["143324"]  # type: ignore
    if "Set-Cookie" in response["headers"]:
        response["headers"]["Set-Cookie"] = [  # type: ignore
            "__cf_bm=fake;path=/; expires=Tue, 15-Oct-24 23:22:45 GMT; domain=.api.openai.com; HttpOnly;Secure; SameSite=None",
            "_cfuvid=fake;path=/; domain=.api.openai.com; HttpOnly; Secure; SameSite=None",
        ]  # type: ignore
    if "set-cookie" in response["headers"]:
        response["headers"]["set-cookie"] = [  # type: ignore
            "guest_id_marketing=v1%3FAKEBROTHER; Max-Age=63072000; Expires=Sat, 19 Dec 2026 19:52:20 GMT; Path=/; Domain=.x.com; Secure; SameSite=None",
            "guest_id_ads=v1%3FAKEBROTHER; Max-Age=63072000; Expires=Sat, 19 Dec 2026 19:52:20 GMT; Path=/; Domain=.x.com; Secure; SameSite=None",
            "personalization_id=v1_SUPERFAKE; Max-Age=63072000; Expires=Sat, 19 Dec 2026 19:52:20 GMT; Path=/; Domain=.x.com; Secure; SameSite=None",
            "guest_id=v1%3FAKEBROTHER; Max-Age=63072000; Expires=Sat, 19 Dec 2026 19:52:20 GMT; Path=/; Domain=.x.com; Secure; SameSite=None",
        ]  # type: ignore

    return response


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

    @pytest.mark.skip_until(
        deadline=datetime(2025, 1, 25),
        strict=True,
        msg="Still figuring out how to tech llm's to test with dpytest",
    )
    @pytest.mark.flaky(retries=3, delay=5)
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

    @pytest.mark.vcronly()
    @pytest.mark.gallerydlonly()
    @pytest.mark.default_cassette("test_cli_oauth_failure.yaml")
    @pytest.mark.vcr(
        record_mode="new_episodes",
        allow_playback_repeats=True,
        match_on=["scheme", "port", "path"],  # Removed method and query to be more lenient
        ignore_localhost=False,
        before_record_response=_filter_response,
        before_record_request=_filter_request_headers,
        # before_record_request=filter_sensitive_data,
        filter_post_data_parameters=[("client_id", "FILTERED"), ("client_secret", "FILTERED")],
        filter_query_parameters=["client_id", "client_secret"],
    )
    def test_cli_oauth_failure(
        self, vcr: VCRRequest, caplog: LogCaptureFixture, capsys: CaptureFixture, mocker: MockerFixture
    ) -> None:
        """Test OAuth flow failure."""
        # Mock the OAuth flow

        # import bpdb; bpdb.set_trace()
        with capsys.disabled():
            with ContextLogger(caplog) as _logger:
                _logger.add(sys.stdout, level="DEBUG")
                caplog.set_level(logging.DEBUG)

                with tracing_context(enabled=False):
                    mock_flow = mocker.MagicMock()
                    mock_flow.start.return_value = "https://dropbox.com/oauth"
                    mock_flow.finish.side_effect = Exception("OAuth failed")

                    mocker.patch("dropbox.DropboxOAuth2FlowNoRedirect", return_value=mock_flow)
                    mocker.patch("builtins.input", return_value="test_code")
                    mock_print = mocker.patch("builtins.print")
                    mock_exit = mocker.patch("sys.exit")

                    # Execute
                    cli_oauth()

                    # Verify error handling
                    mock_print.assert_any_call(
                        "Error: 400 Client Error: Bad Request for url: https://api.dropboxapi.com/oauth2/token"
                    )
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

        result = await get_dropbox_client(oauth2_access_token="test_token")  # noqa: S106
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
