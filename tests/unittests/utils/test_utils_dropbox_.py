# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import asyncio
import os

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pytest_asyncio
import requests

import pytest

from pytest_mock import MockerFixture

from democracy_exe import aio_settings
from democracy_exe.utils import dropbox_
from democracy_exe.utils.dropbox_ import BadInputException, create_session
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
EXPIRATION = datetime.utcnow() + timedelta(seconds=EXPIRES_IN)

EXPIRATION_BUFFER = timedelta(minutes=5)

# pylint: disable=protected-access

is_running_in_github = bool(os.environ.get("GITHUB_ACTOR"))


# TODO: mock this correctly
@pytest.mark.dropboxonly
@pytest.mark.unittest
class TestDropboxClient:
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
