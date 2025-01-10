"""democracy_exe.utils.dropbox_"""
# pylint: disable=unused-import
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
# pylint: disable=no-member
from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
import os
import pathlib
import sys
import time
import traceback
import unicodedata

from collections import defaultdict
from collections.abc import Generator
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import aiofiles
import dropbox
import requests
import rich
import six
import structlog

from dropbox import DropboxOAuth2FlowNoRedirect, create_session
from dropbox.dropbox_client import BadInputException
from dropbox.exceptions import ApiError, AuthError
from dropbox.files import FileMetadata, FolderMetadata, WriteMode
from dropbox.users import Account


logger = structlog.get_logger(__name__)
from pydantic import Field, SecretStr

from democracy_exe.aio_settings import aiosettings


async def get_dropbox_session() -> requests.Session:
    """Get a new Dropbox session asynchronously.

    Returns:
        requests.Session: A configured session for Dropbox API requests.
    """
    return await asyncio.to_thread(create_session)


async def get_dropbox_client(oauth2_access_token: str | None = None) -> dropbox.Dropbox | None:  # pylint: disable=no-member
    """Create and initialize a Dropbox client.

    Args:
        oauth2_access_token: Optional OAuth2 access token. If not provided, uses token from settings.

    Returns:
        dropbox.Dropbox | None: An authenticated Dropbox client or None if initialization fails.

    Raises:
        BadInputException: If the provided token is invalid.
        AuthError: If authentication fails.
        Exception: For other unexpected errors.
    """
    dbx = None
    try:
        if oauth2_access_token is None:
            # token_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_token)
            oauth2_access_token = aiosettings.dropbox_cerebro_token.get_secret_value()  # pylint: disable=no-member
        dbx = dropbox.Dropbox(oauth2_access_token=oauth2_access_token)
        # Run potentially blocking operation in thread pool
        await asyncio.to_thread(dbx.users_get_current_account)
        logger.info("Connected to Dropbox successfully")
    except BadInputException as ex:
        logger.error(f"Error Class: {ex.__class__!s}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        logger.warning(output)
        logger.error(f"exc_type: {sys.exc_info()[0]}")
        logger.error(f"exc_value: {sys.exc_info()[1]}")
        traceback.print_tb(sys.exc_info()[2])
        raise
    except AuthError:
        sys.exit("ERROR: Invalid access token; try re-generating an access token from the app console on the web.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

    return dbx


async def list_files_in_remote_folder(dbx: dropbox.Dropbox) -> None:
    """List all files in a remote Dropbox folder asynchronously.

    Args:
        dbx: An authenticated Dropbox client.

    Raises:
        Exception: If listing files fails.
    """
    logger.info("Listing files in remote folder")
    try:
        folder_path = aiosettings.default_dropbox_folder
        # Run potentially blocking operation in thread pool
        files = await asyncio.to_thread(
            lambda: dbx.files_list_folder(folder_path, recursive=True).entries
        )
        rich.print("------------Listing Files in Folder------------ ")

        for file in files:
            rich.print(file.name)
        logger.info("Successfully listed files")
        # await logger.complete()

    except Exception as ex:
        logger.error(f"Error Class: {ex.__class__!s}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        logger.warning(output)
        logger.error(f"exc_type: {sys.exc_info()[0]}")
        logger.error(f"exc_value: {sys.exc_info()[1]}")
        traceback.print_tb(sys.exc_info()[2])
        # await logger.complete()
        raise


async def download_img(dbx: dropbox.Dropbox, short_file_name: str) -> None:
    """Download an image file from Dropbox asynchronously.

    Args:
        dbx: An authenticated Dropbox client.
        short_file_name: The name of the file to download.

    Raises:
        Exception: If downloading or writing the file fails.
    """
    try:
        # Run potentially blocking download in thread pool
        metadata, res = await asyncio.to_thread(
            dbx.files_download,
            path=f"{aiosettings.default_dropbox_folder}/{short_file_name}"
        )

        async with aiofiles.open(f"{short_file_name}", "wb") as f:
            await f.write(res.content)

    except Exception as ex:
        logger.error(f"Error Class: {ex.__class__!s}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        logger.warning(output)
        logger.error(f"exc_type: {sys.exc_info()[0]}")
        logger.error(f"exc_value: {sys.exc_info()[1]}")
        traceback.print_tb(sys.exc_info()[2])
        raise


@contextlib.contextmanager
def stopwatch(message: str) -> Generator[None, None, None]:
    """Context manager to print how long a block of code took.

    Args:
        message: Message to display with the elapsed time.

    Yields:
        Generator[None, None, None]: A context manager that times code execution.
    """
    t0 = time.time()
    try:
        yield
    finally:
        t1 = time.time()
        print(f"Total elapsed time for {message}: {t1 - t0:.3f}")


async def list_folder(dbx: dropbox.Dropbox, folder: str, subfolder: str) -> dict[str, FileMetadata | FolderMetadata]:
    """List a folder asynchronously.

    Args:
        dbx: An authenticated Dropbox client.
        folder: The root folder to list.
        subfolder: The subfolder within the root folder.

    Returns:
        Dict[str, Union[FileMetadata, FolderMetadata]]: A dictionary mapping filenames to their metadata.

    Raises:
        dropbox.exceptions.ApiError: If folder listing fails.
    """
    logger.info(f"Listing folder: /{folder}/{subfolder}")
    path = f'/{folder}/{subfolder.replace(os.path.sep, "/")}'
    while "//" in path:
        path = path.replace("//", "/")
    path = path.rstrip("/")
    try:
        with stopwatch("list_folder"):
            res = await asyncio.to_thread(dbx.files_list_folder, path)
        logger.info("Successfully listed folder contents")
        # await logger.complete()
        return {entry.name: entry for entry in res.entries}
    except dropbox.exceptions.ApiError as err:
        logger.error(f"Folder listing failed for {path}: {err}")
        # await logger.complete()
        return {}


async def download(
    dbx: dropbox.Dropbox,
    folder: str,
    subfolder: str,
    name: str
) -> bytes | None:
    """Download a file asynchronously.

    Args:
        dbx: An authenticated Dropbox client.
        folder: The root folder containing the file.
        subfolder: The subfolder within the root folder.
        name: The name of the file to download.

    Returns:
        Optional[bytes]: The file contents as bytes, or None if download fails.

    Raises:
        dropbox.exceptions.HttpError: If HTTP request fails.
    """
    logger.info(f"Downloading file: {name} from /{folder}/{subfolder}")
    path = f'/{folder}/{subfolder.replace(os.path.sep, "/")}/{name}'
    while "//" in path:
        path = path.replace("//", "/")
    with stopwatch("download"):
        try:
            md, res = await asyncio.to_thread(dbx.files_download, path)
            data = res.content
            logger.info(f"Successfully downloaded {len(data)} bytes")
            # await logger.complete()
            return data
        except dropbox.exceptions.HttpError as err:
            logger.error(f"HTTP error during download: {err}")
            # await logger.complete()
            return None


async def upload(
    dbx: dropbox.Dropbox,
    fullname: str,
    folder: str,
    subfolder: str,
    name: str,
    overwrite: bool = False
) -> FileMetadata | None:
    """Upload a file asynchronously.

    Args:
        dbx: An authenticated Dropbox client.
        fullname: The full local path to the file.
        folder: The root folder to upload to.
        subfolder: The subfolder within the root folder.
        name: The name to give the file in Dropbox.
        overwrite: Whether to overwrite existing files. Defaults to False.

    Returns:
        Optional[FileMetadata]: The uploaded file's metadata, or None if upload fails.

    Raises:
        OSError: If file operations fail.
        dropbox.exceptions.ApiError: If upload fails.
    """
    path = f'/{folder}/{subfolder.replace(os.path.sep, "/")}/{name}'
    while "//" in path:
        path = path.replace("//", "/")
    mode = WriteMode.overwrite if overwrite else WriteMode.add
    mtime = await asyncio.to_thread(os.path.getmtime, fullname)

    async with aiofiles.open(fullname, "rb") as f:
        data = await f.read()

    with stopwatch("upload %d bytes" % len(data)):
        try:
            res = await asyncio.to_thread(
                dbx.files_upload,
                data,
                path,
                mode,
                client_modified=datetime.datetime(*time.gmtime(mtime)[:6]),
                mute=True,
            )
            return res
        except dropbox.exceptions.ApiError as err:
            logger.error(f"API error: {err}")
            return None


async def iter_dir_and_upload(dbx: dropbox.Dropbox, remote_folder: str, local_folder: str) -> None:
    """Iterate through a local directory and upload files to Dropbox asynchronously.

    Args:
        dbx: An authenticated Dropbox client.
        remote_folder: The remote folder path in Dropbox.
        local_folder: The local folder path to upload from.

    Raises:
        Exception: If directory iteration or upload fails.
    """
    try:
        # Run os.walk in thread pool since it's blocking
        walk_results = await asyncio.to_thread(lambda: list(os.walk(local_folder)))

        for root, _dirs, files in walk_results:
            for filename in files:
                local_path = os.path.join(root, filename)
                relative_path = os.path.relpath(local_path, local_folder)
                dropbox_path = os.path.join(remote_folder, relative_path)

                async with aiofiles.open(local_path, 'rb') as f:
                    file_data = await f.read()
                    await asyncio.to_thread(
                        dbx.files_upload,
                        file_data,
                        dropbox_path,
                        mode=WriteMode('overwrite')
                    )

    except Exception as ex:
        logger.error(f"Error Class: {ex.__class__!s}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        logger.warning(output)
        logger.error(f"exc_type: {sys.exc_info()[0]}")
        logger.error(f"exc_value: {sys.exc_info()[1]}")
        traceback.print_tb(sys.exc_info()[2])
        raise


async def co_upload_to_dropbox(
    dbx: dropbox.Dropbox,
    path_to_local_file: str,
    path_to_remote_dir: str | None = None
) -> None:
    """Upload a file to Dropbox asynchronously.

    Args:
        dbx: An authenticated Dropbox client.
        path_to_local_file: The local file path to upload.
        path_to_remote_dir: Optional remote directory path. If not provided, uses default.

    Raises:
        Exception: If file upload fails.
    """
    try:
        if path_to_remote_dir is None:
            path_to_remote_dir = aiosettings.default_dropbox_folder

        file_name = os.path.basename(path_to_local_file)
        dropbox_path = f"{path_to_remote_dir}/{file_name}"

        async with aiofiles.open(path_to_local_file, mode='rb') as f:
            contents = await f.read()
            dbx.files_upload(contents, dropbox_path, mode=WriteMode('overwrite'))

    except Exception as ex:
        logger.error(f"Error Class: {ex.__class__!s}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        logger.warning(output)
        logger.error(f"exc_type: {sys.exc_info()[0]}")
        logger.error(f"exc_value: {sys.exc_info()[1]}")
        traceback.print_tb(sys.exc_info()[2])
        raise


def select_revision(
    dbx: dropbox.Dropbox,
    filename: str | None = None,
    path_to_remote_file_or_dir: str | None = None
) -> str:
    """Select a revision of a file in Dropbox.

    Args:
        dbx: An authenticated Dropbox client.
        filename: Optional filename to select revision for.
        path_to_remote_file_or_dir: Optional remote path. If not provided, uses default.

    Returns:
        str: The selected revision path.

    Raises:
        Exception: If revision selection fails.
    """
    try:
        if path_to_remote_file_or_dir is None:
            path_to_remote_file_or_dir = aiosettings.default_dropbox_folder

        if filename is None:
            return path_to_remote_file_or_dir

        return f"{path_to_remote_file_or_dir}/{filename}"

    except Exception as ex:
        logger.error(f"Error Class: {ex.__class__!s}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        logger.warning(output)
        logger.error(f"exc_type: {sys.exc_info()[0]}")
        logger.error(f"exc_value: {sys.exc_info()[1]}")
        traceback.print_tb(sys.exc_info()[2])
        raise


def cli_oauth() -> None:
    """Walk through a basic oauth flow using the existing long-lived token type.
    Populate your app key and app secret in order to run this locally.

    Raises:
        SystemExit: If OAuth flow fails.
    """
    # app_key_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_app_key)
    # app_secret_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_app_secret)

    auth_flow = DropboxOAuth2FlowNoRedirect(
        aiosettings.dropbox_cerebro_app_key.get_secret_value(),
        aiosettings.dropbox_cerebro_app_secret.get_secret_value()
    )

    authorize_url = auth_flow.start()
    print(f"1. Go to: {authorize_url}")
    print('2. Click "Allow" (you might have to log in first).')
    print("3. Copy the authorization code.")
    auth_code = input("Enter the authorization code here: ").strip()

    try:
        oauth_result = auth_flow.finish(auth_code)
        with dropbox.Dropbox(oauth2_access_token=oauth_result.access_token) as dbx:
            dbx.users_get_current_account()
            rich.print("Successfully set up client!")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


class AsyncDropBox:
    """Asynchronous Dropbox client implementation."""

    def __init__(self) -> None:
        """Initialize AsyncDropBox with credentials from settings."""
        # DISABLED: This gets around pyright/pylance but exposes the secret values to the global namespace.
        # token_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_token)
        # app_key_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_app_key)
        # app_secret_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_app_secret)
        # refresh_token_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_token)

        # self.dropbox_access_token: str = token_secret.get_secret_value()
        # self.app_key: str = app_key_secret.get_secret_value()
        # self.app_secret: str = app_secret_secret.get_secret_value()
        # self.dropbox_refresh_token: str = refresh_token_secret.get_secret_value()

        # token_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_token)
        # app_key_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_app_key)
        # app_secret_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_app_secret)
        # refresh_token_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_token)

        self.dropbox_access_token: str = aiosettings.dropbox_cerebro_token.get_secret_value()  # pylint: disable=no-member
        self.app_key: str = aiosettings.dropbox_cerebro_app_key.get_secret_value()  # pylint: disable=no-member
        self.app_secret: str = aiosettings.dropbox_cerebro_app_secret.get_secret_value()  # pylint: disable=no-member
        self.dropbox_refresh_token: str = aiosettings.dropbox_cerebro_token.get_secret_value()  # pylint: disable=no-member

        self.client = self.auth()

    async def __aenter__(self) -> AsyncDropBox:
        """Async context manager entry.

        Returns:
            AsyncDropBox: The instance itself.
        """
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit.

        Args:
            *args: Variable length argument list.
        """
        self.client.close()

    def auth(self) -> dropbox.Dropbox:
        """Authenticate with Dropbox.

        Authentication is done via OAuth2. You can generate token for yourself in the App Console.
        See https://blogs.dropbox.com/developers/2014/05/generate-an-access-token-for-your-own-account/
        Authentication step initially is done with ACCESS_TOKEN, as it is short lived it will expire soon.
        Therefore better to have Refresh token.

        Returns:
            dropbox.Dropbox: An authenticated Dropbox client.
        """
        return (
            dropbox.Dropbox(
                oauth2_access_token=self.dropbox_access_token,
                oauth2_refresh_token=self.dropbox_refresh_token,
                app_key=self.app_key,
                app_secret=self.app_secret,
            )
            if self.dropbox_refresh_token
            else dropbox.Dropbox(self.dropbox_access_token)
        )

    async def account_info(self) -> dict[str, Any]:
        """Get account information of the current user.

        Returns:
            Dict[str, Any]: A dictionary containing account information.
        """
        temp = defaultdict(dict)
        result = self.client.users_get_current_account()

        temp["abbreviated_name"] = result.name.abbreviated_name
        temp["display_name"] = result.name.display_name
        temp["familiar_name"] = result.name.familiar_name
        temp["given_name"] = result.name.given_name
        temp["surname"] = result.name.surname

        temp["account_id"] = result.account_id
        temp["country"] = result.country
        temp["disabled"] = result.disabled
        temp["email"] = result.email
        temp["email_verified"] = result.email_verified
        temp["is_paired"] = result.is_paired
        temp["locale"] = result.locale

        temp["profile_photo_url"] = result.profile_photo_url
        temp["referral_link"] = result.referral_link
        temp["team"] = result.team
        temp["team_member_id"] = result.team_member_id
        return temp

    async def list_files(
        self,
        path: str,
        recursive: bool = False,
        include_media_info: bool = False,
        include_deleted: bool = False,
        include_has_explicit_shared_members: bool = False,
        include_mounted_folders: bool = True,
        limit: int | None = None,
        shared_link: str | None = None,
        include_property_groups: list[str] | None = None,
        include_non_downloadable_files: bool = True,
    ) -> dict[str, str]:
        """List files in a Dropbox folder.

        Args:
            path (str): Path to list files from.
            recursive (bool, optional): Whether to list recursively. Defaults to False.
            include_media_info (bool, optional): Include media info. Defaults to False.
            include_deleted (bool, optional): Include deleted files. Defaults to False.
            include_has_explicit_shared_members (bool, optional): Include explicit shared members. Defaults to False.
            include_mounted_folders (bool, optional): Include mounted folders. Defaults to True.
            limit (Optional[int], optional): Maximum number of results. Defaults to None.
            shared_link (Optional[str], optional): Shared link to use. Defaults to None.
            include_property_groups (Optional[List[str]], optional): Property groups to include. Defaults to None.
            include_non_downloadable_files (bool, optional): Include non-downloadable files. Defaults to True.

        Returns:
            Dict[str, str]: A dictionary mapping file paths to their shared links.
        """
        response = self.client.files_list_folder(
            path=path,
            recursive=recursive,
            include_media_info=include_media_info,
            include_deleted=include_deleted,
            include_has_explicit_shared_members=include_has_explicit_shared_members,
            include_mounted_folders=include_mounted_folders,
            limit=limit,
            shared_link=shared_link,
            include_property_groups=include_property_groups,
            include_non_downloadable_files=include_non_downloadable_files,
        )
        temp: dict[str, str] = {}

        try:
            for file in response.entries:
                link = self.client.sharing_create_shared_link(file.path_display)
                path = link.url.replace("0", "1")
                temp[file.path_display] = path
            return temp
        except Exception as er:
            print(er)
            return temp

    async def upload_file(self, file_from: str, file_to: str) -> None:
        """Upload a file to Dropbox.

        Args:
            file_from (str): Local file path.
            file_to (str): Remote file path.
        """
        with open(file_from, "rb") as f:
            self.client.files_upload(f.read(), file_to, mode=WriteMode("overwrite"))

    async def save_file_localy(self, file_path: str, filename: str) -> None:
        """Save a file locally from Dropbox.

        Args:
            file_path (str): Remote file path.
            filename (str): Remote filename.
        """
        metadata, res = self.client.files_download(file_path + filename)
        with open(metadata.name, "wb") as f:
            f.write(res.content)

    async def get_link_of_file(self, file_path: str, filename: str, dowload: bool = False) -> dict[str, str]:
        """Get a shared link for a file.

        Args:
            file_path (str): Remote file path.
            filename (str): Remote filename.
            dowload (bool, optional): Whether to get a download link. Defaults to False.

        Returns:
            Dict[str, str]: A dictionary containing the file URL.
        """
        path = self.client.sharing_create_shared_link(file_path + filename)
        if dowload:
            path = path.url.replace("0", "1")
        return {"file": path.url}


async def co_upload_to_dropbox2(path_to_local_file: str, path_to_remote_dir: str = "/") -> None:
    """Upload a file to Dropbox using the async client.

    Args:
        path_to_local_file (str): Path to the local file to upload.
        path_to_remote_dir (str, optional): Remote directory to upload to. Defaults to "/".

    Raises:
        AssertionError: If the local file doesn't exist or is not a file.
        ApiError: If there's an error during upload, including insufficient space.
    """
    localfile_pathobj = pathlib.Path(f"{path_to_local_file}").absolute()
    try:
        assert localfile_pathobj.exists()
        assert localfile_pathobj.is_file()
    except Exception as ex:
        print(ex)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.error(f"Error Class: {ex.__class__!s}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        logger.warning(output)
        logger.error(f"exc_type: {exc_type}")
        logger.error(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        raise

    _localfile = f"{localfile_pathobj}"
    _backuppath = f"{path_to_remote_dir}/{path_to_local_file}"

    try:
        async with AsyncDropBox() as drop:
            await drop.upload_file(file_from=_localfile, file_to=_backuppath)
    except ApiError as err:
        if err.error.is_path() and err.error.get_path().reason.is_insufficient_space():
            logger.error("ERROR: Cannot back up; insufficient space.")
        elif err.user_message_text:
            rich.print(err.user_message_text)

        else:
            rich.print(err)
