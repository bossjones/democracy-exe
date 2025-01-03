# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false


from __future__ import annotations

import warnings

from _pytest.fixtures import FixtureRequest


warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

import datetime
import os
import random
import string
import sys

from io import BytesIO

import dropbox

from dropbox.files import DeleteResult

import pytest

from democracy_exe.utils import dropbox_


# Key Types
REFRESH_TOKEN_KEY = "REFRESH_TOKEN"
ACCESS_TOKEN_KEY = "TOKEN"
CLIENT_ID_KEY = "APP_KEY"
CLIENT_SECRET_KEY = "APP_SECRET"
# App Types
# SCOPED_KEY = "DROPBOX"
SCOPED_KEY = "DEMOCRACY_EXE_CONFIG_DROPBOX"
LEGACY_KEY = "LEGACY"
# User Types
USER_KEY = "CEREBRO"
TEAM_KEY = "TEAM"
# Misc types
SHARED_LINK_KEY = "DROPBOX_SHARED_LINK"


def format_env_name(
    app_type: str = SCOPED_KEY,
    user_type: str = USER_KEY,
    key_type: str = ACCESS_TOKEN_KEY,
) -> str:
    # 'DROPBOX_CEREBRO_TOKEN'
    return f"{app_type}_{user_type}_{key_type}"


def _value_from_env_or_die(env_name: str) -> str:
    value = os.environ.get(env_name)
    if value is None:
        print(
            f"Set {env_name} environment variable to a valid value.",
            file=sys.stderr,
        )
        sys.exit(1)
    return value


@pytest.fixture()
def dbx_from_env() -> dropbox.Dropbox:
    oauth2_token = _value_from_env_or_die(format_env_name())
    return dropbox_.get_dropbox_client(oauth2_access_token=oauth2_token)


MALFORMED_TOKEN = "asdf"
INVALID_TOKEN = "z" * 62

# Need bytes type for Python3
DUMMY_PAYLOAD = string.ascii_letters.encode("ascii")

RANDOM_FOLDER = random.sample(string.ascii_letters, 15)
TIMESTAMP = str(datetime.datetime.now(datetime.UTC))
STATIC_FILE = "/test.txt"


@pytest.fixture(scope="module", autouse=True)
def pytest_setup() -> None:
    print("Setup")
    dbx = dropbox_.get_dropbox_client(oauth2_access_token=_value_from_env_or_die(format_env_name()))

    try:
        dbx.files_delete_v2(STATIC_FILE)  # type: ignore
    except Exception:
        print(f"File not found in dropbox remote -> {STATIC_FILE}")

    try:
        dbx.files_delete_v2(f"/Test/{TIMESTAMP}")
    except Exception:
        print(f"File not found in dropbox remote -> /Test/{TIMESTAMP}")


# pylint: disable=protected-access

if os.environ.get("GITHUB_ACTOR"):
    is_running_in_github = True
else:
    is_running_in_github = False


@pytest.mark.filterwarnings("ignore:unclosed <ssl.SSLSocket ")
@pytest.mark.dropboxonly
@pytest.mark.integration
@pytest.mark.usefixtures(
    "dbx_from_env",
)
class TestDropboxIntegration:
    # def test_rpc(self, dbx_from_env: FixtureRequest) -> None:
    #     dbx_from_env.files_list_folder("")

    #     # Test API error
    #     random_folder_path = "/" + "".join(RANDOM_FOLDER)
    #     with pytest.raises(ApiError) as cm:
    #         dbx_from_env.files_list_folder(random_folder_path)
    #     assert isinstance(cm.value.error, ListFolderError)

    @pytest.mark.flaky(reruns=5, reruns_delay=2)
    def test_upload_download(self, dbx_from_env: FixtureRequest) -> None:
        # Upload file
        random_filename = "".join(RANDOM_FOLDER)
        random_path = f"/Test/{TIMESTAMP}/{random_filename}"
        test_contents = DUMMY_PAYLOAD
        dbx_from_env.files_upload(test_contents, random_path)  # type: ignore

        # Download file
        _, resp = dbx_from_env.files_download(random_path)  # type: ignore
        assert resp.content == DUMMY_PAYLOAD

        # Cleanup folder
        dbx_from_env.files_delete_v2(f"/Test/{TIMESTAMP}")

    def test_bad_upload_types(self, dbx_from_env: FixtureRequest) -> None:
        with pytest.raises(TypeError):
            dbx_from_env.files_upload(BytesIO(b"test"), "/Test")  # type: ignore

    def test_clone_when_user_linked(self, dbx_from_env: FixtureRequest) -> None:
        new_dbx = dbx_from_env.clone()  # type: ignore
        assert dbx_from_env is not new_dbx
        assert isinstance(new_dbx, dbx_from_env.__class__)

    def test_versioned_route(self, dbx_from_env: FixtureRequest) -> None:
        # Upload a test file
        dbx_from_env.files_upload(DUMMY_PAYLOAD, STATIC_FILE)  # type: ignore

        # Delete the file with v2 route
        resp = dbx_from_env.files_delete_v2(STATIC_FILE)  # type: ignore
        # Verify response type is of v2 route
        assert isinstance(resp, DeleteResult)
