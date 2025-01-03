#!/usr/bin/env python
from __future__ import annotations

import os

from datetime import datetime, timedelta

import requests

import pytest

from pytest_mock import MockerFixture

from democracy_exe.utils import dropbox_
from democracy_exe.utils.dropbox_ import BadInputException, create_session


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

    def test_default_Dropbox_raises_assertion_error(self) -> None:
        with pytest.raises(BadInputException):
            # Requires either access token or refresh token
            dropbox_.get_dropbox_client(token="")
