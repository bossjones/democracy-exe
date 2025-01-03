"""democracy_exe.utils.aiodbx.

Provides asynchronous Dropbox API client functionality.
"""

# SOURCE: https://github.com/ebai101/aiodbx/blob/main/aiodbx.py
from __future__ import annotations

import asyncio
import base64
import json
import os
import pathlib
import sys
import traceback

from collections.abc import AsyncGenerator, Callable, Generator
from typing import Any, Dict, List, Optional, Union

import aiofiles
import aiohttp

from loguru import logger

from democracy_exe.aio_settings import aiosettings


class DropboxAPIError(Exception):
    """Exception for errors thrown by the API.

    Contains the HTTP status code and the returned error message.

    Args:
        status: HTTP status code
        message: Error message from API response

    Attributes:
        status: HTTP status code
        message: Error message
    """

    def __init__(self, status: int, message: str | dict[str, Any]) -> None:
        self.status = status
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        if not isinstance(self.message, str):
            return f"{self.status} {self.message}"
        try:
            self.message = json.loads(self.message)
            return f'{self.status} {self.message["error_summary"]}'
        except Exception:
            return f"{self.status} {self.message}"


class Request:
    """Wrapper for a ClientResponse object that allows automatic retries for certain statuses.

    Args:
        request: Session method to call that returns a ClientResponse (e.g. session.post)
        url: URL to request
        ok_statuses: List of statuses that will return without an error (default is [200])
        retry_count: Number of times that the request will be retried (default is 5)
        retry_statuses: List of statuses that will cause automatic retry (default is [429])
        **kwargs: Arbitrary keyword arguments passed to request callable

    Attributes:
        request: The request callable
        url: Target URL
        ok_statuses: List of acceptable status codes
        retry_count: Maximum retry attempts
        retry_statuses: Status codes that trigger retry
        kwargs: Additional request parameters
        current_attempt: Current retry attempt number
        resp: The current response object
    """

    def __init__(
        self,
        request: Callable[..., Any],
        url: str,
        ok_statuses: list[int] | None = None,
        retry_count: int = 5,
        retry_statuses: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        if ok_statuses is None:
            ok_statuses = [200]
        if retry_statuses is None:
            retry_statuses = [429]
        self.request = request
        self.url = url
        self.ok_statuses = ok_statuses
        self.retry_count = retry_count
        self.retry_statuses = retry_statuses
        self.kwargs = kwargs
        self.trace_request_ctx = kwargs.pop("trace_request_ctx", {})

        self.current_attempt = 0
        self.resp: aiohttp.ClientResponse | None = None

    async def _do_request(self) -> aiohttp.ClientResponse:
        """Performs a request with automatic retries for specific return statuses.

        This internal method handles the actual HTTP request execution and implements
        the retry logic. It will automatically retry requests that return status codes
        specified in retry_statuses, up to retry_count times.

        Should not be called directly. Instead, use an `async with` block with a Request
        object to manage the response context properly.

        Returns:
            aiohttp.ClientResponse: The response from the request. If a retry occurs,
                returns the response from the last successful attempt.

        Raises:
            DropboxAPIError: If response status is >= 400 and not in ok_statuses.
                The error will contain the HTTP status code and response text.

        Note:
            If a retry-able status code is received and Retry-After header is present,
            the retry will wait for the specified duration. Otherwise, it defaults to
            1 second between retries.
        """
        self.current_attempt += 1
        if self.current_attempt > 1:
            logger.debug(f"Attempt {self.current_attempt} out of {self.retry_count}")

        resp: aiohttp.ClientResponse = await self.request(
            self.url,
            **self.kwargs,
            trace_request_ctx={
                "current_attempt": self.current_attempt,
                **self.trace_request_ctx,
            },
        )

        if resp.status not in self.ok_statuses and resp.status >= 400:
            raise DropboxAPIError(resp.status, await resp.text())

        endpoint_name = self.url[self.url.index("2") + 1 :]
        logger.debug(f"Request OK: {endpoint_name} returned {resp.status}")
        if self.current_attempt < self.retry_count and resp.status in self.retry_statuses:
            if "Retry-After" in resp.headers:
                sleep_time = int(resp.headers["Retry-After"])
            else:
                sleep_time = 1
            await asyncio.sleep(sleep_time)
            return await self._do_request()

        self.resp = resp
        await logger.complete()
        return resp

    def __await__(self) -> Generator[Any, None, aiohttp.ClientResponse]:
        """Makes the Request class awaitable.

        This allows the Request object to be used with the await keyword directly,
        which will execute the request and return the response.

        Returns:
            Generator[Any, None, aiohttp.ClientResponse]: A generator that yields the
                ClientResponse when awaited.

        Example:
            response = await Request(session.post, "https://api.example.com")
        """
        return self.__aenter__().__await__()

    async def __aenter__(self) -> aiohttp.ClientResponse:
        """Async context manager entry point.

        This method is called when entering an async context manager block using
        'async with'. It executes the request and returns the response.

        Returns:
            aiohttp.ClientResponse: The response from the executed request.

        Example:
            async with Request(session.post, "https://api.example.com") as response:
                data = await response.json()
        """
        result = await self._do_request()
        await logger.complete()
        return result

    async def __aexit__(self, *excinfo: Any) -> None:
        """Async context manager exit point.

        This method is called when exiting an async context manager block. It ensures
        proper cleanup of resources by closing the response if it exists and hasn't
        been closed already.

        Args:
            *excinfo: Exception information if an error occurred in the context manager
                block. Contains (exc_type, exc_value, traceback) or None if no exception.

        Note:
            This method is automatically called when exiting an 'async with' block
            and handles cleanup even if an exception occurred in the block.
        """
        if self.resp is not None and not self.resp.closed:
            self.resp.close()
        await logger.complete()


class AsyncDropboxAPI:
    """Dropbox API client using asynchronous HTTP requests.

    Args:
        token: Dropbox API access token
        retry_statuses: List of statuses that will automatically be retried (default is [429])

    Attributes:
        token: The API access token
        retry_statuses: Status codes that trigger retry
        client_session: Aiohttp client session
        upload_session: List of upload session entries
    """

    def __init__(self, token: str, retry_statuses: list[int] | None = None) -> None:
        if retry_statuses is None:
            retry_statuses = [429]
        self.token = token
        self.retry_statuses = retry_statuses
        self.client_session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit_per_host=50))
        self.upload_session: list[dict[str, Any]] = []

    async def validate(self) -> bool:
        """Validates the user authentication token.

        See: https://www.dropbox.com/developers/documentation/http/documentation#check-user

        Returns:
            True if the API returns the same string (token is valid)

        Raises:
            DropboxAPIError: If the token is invalid
        """
        logger.debug("Validating token")

        nonce = base64.b64encode(os.urandom(8), altchars=b"-_").decode("utf-8")
        url = "https://api.dropboxapi.com/2/check/user"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = json.dumps({"query": nonce})

        async with Request(self.client_session.post, url, headers=headers, data=data) as resp:
            resp_data = await resp.json()
            if resp_data["result"] != nonce:
                raise DropboxAPIError(resp.status, "Token is invalid")
            logger.debug("Token is valid")
            await logger.complete()
            return True

    async def download_file(self, dropbox_path: str, local_path: str | None = None) -> str:
        """Downloads a single file.

        See: https://www.dropbox.com/developers/documentation/http/documentation#files-download

        Args:
            dropbox_path: File path on Dropbox to download from
            local_path: Path on local disk to download to (defaults to current directory)

        Returns:
            Local path where file was downloaded to
        """
        # default to current directory
        if local_path is None:
            local_path = os.path.basename(dropbox_path)

        logger.info(f"Downloading {os.path.basename(local_path)}")
        logger.debug(f"from {dropbox_path}")

        url = "https://content.dropboxapi.com/2/files/download"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Dropbox-API-Arg": json.dumps({"path": dropbox_path}),
        }

        async with Request(self.client_session.post, url, headers=headers) as resp:
            async with aiofiles.open(local_path, "wb") as f:
                async for chunk, _ in resp.content.iter_chunks():
                    await f.write(chunk)

                await logger.complete()
                return local_path

    async def download_folder(self, dropbox_path: str, local_path: str | None = None) -> str:
        """Downloads a folder as a zip file.

        See: https://www.dropbox.com/developers/documentation/http/documentation#files-download_zip

        Args:
            dropbox_path: Folder path on Dropbox to download from
            local_path: Path on local disk to download to (defaults to current directory)

        Returns:
            Local path where zip file was downloaded to
        """
        # default to current directory
        if local_path is None:
            local_path = os.path.basename(dropbox_path)

        logger.info(f"Downloading {os.path.basename(local_path)}")
        logger.debug(f"from {dropbox_path}")

        url = "https://content.dropboxapi.com/2/files/download_zip"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Dropbox-API-Arg": json.dumps({"path": dropbox_path}),
        }

        async with Request(self.client_session.post, url, headers=headers) as resp:
            async with aiofiles.open(local_path, "wb") as f:
                async for chunk, _ in resp.content.iter_chunks():
                    await f.write(chunk)

                await logger.complete()
                return local_path

    async def download_shared_link(self, shared_link: str, local_path: str | None = None) -> str:
        """Downloads a file from a shared link.

        See: https://www.dropbox.com/developers/documentation/http/documentation#sharing-get_shared_link_file

        Args:
            shared_link: Shared link to download from
            local_path: Path on local disk to download to (defaults to current directory)

        Returns:
            Local path where file was downloaded to
        """
        # default to current directory, with the path in the shared link
        if local_path is None:
            local_path = os.path.basename(shared_link[: shared_link.index("?")])

        logger.info(f"Downloading {os.path.basename(local_path)}")
        logger.debug(f"from {shared_link}")

        url = "https://content.dropboxapi.com/2/sharing/get_shared_link_file"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Dropbox-API-Arg": json.dumps({"url": shared_link}),
        }

        async with Request(self.client_session.post, url, headers=headers) as resp:
            async with aiofiles.open(local_path, "wb") as f:
                async for chunk, _ in resp.content.iter_chunks():
                    await f.write(chunk)

                await logger.complete()
                return local_path

    async def upload_start(self, local_path: str, dropbox_path: str) -> dict[str, Any]:
        """Uploads a single file to an upload session.

        This should be used when uploading large quantities of files.
        See: https://www.dropbox.com/developers/documentation/http/documentation#files-upload_session-start

        Args:
            local_path: Local path to upload from
            dropbox_path: Dropbox path to upload to

        Returns:
            UploadSessionFinishArg dict with upload information.
            This dict is automatically stored in `self.upload_session` to be committed later.

        Raises:
            ValueError: If local_path does not exist
            RuntimeError: If current upload session is larger than 1000 files
        """
        if not os.path.exists(local_path):
            raise ValueError(f"local_path {local_path} does not exist")
        if len(self.upload_session) >= 1000:
            raise RuntimeError("upload_session is too large, you must call upload_finish to commit the batch")

        logger.info(f"Uploading {os.path.basename(local_path)}")
        logger.debug(f"to {dropbox_path}")

        url = "https://content.dropboxapi.com/2/files/upload_session/start"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Dropbox-API-Arg": json.dumps({"close": True}),
            "Content-Type": "application/octet-stream",
        }

        async with aiofiles.open(local_path, "rb") as f:
            data = await f.read()
            async with Request(self.client_session.post, url, headers=headers, data=data) as resp:
                resp_data = await resp.json()

                # construct commit entry for finishing batch later
                commit = {
                    "cursor": {
                        "session_id": resp_data["session_id"],
                        "offset": os.path.getsize(local_path),
                    },
                    "commit": {
                        "path": dropbox_path,
                        "mode": "add",
                        "autorename": False,
                        "mute": False,
                    },
                }
                self.upload_session.append(commit)
                await logger.complete()
                return commit

    async def upload_finish(self, check_interval: float = 3) -> list[dict[str, Any]]:
        """Finishes an upload batch.

        See: https://www.dropbox.com/developers/documentation/http/documentation#files-upload_session-finish_batch

        Args:
            check_interval: How often to check upload completion status (default is 3)

        Returns:
            List of FileMetadata dicts containing metadata on each uploaded file

        Raises:
            RuntimeError: If upload_session is empty
            DropboxAPIError: If unknown response is returned from API
        """
        if len(self.upload_session) == 0:
            raise RuntimeError("upload_session is empty, have you uploaded any files yet?")

        logger.info("Finishing upload batch")
        logger.debug(f"Batch size is {len(self.upload_session)}")

        url = "https://api.dropboxapi.com/2/files/upload_session/finish_batch"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = json.dumps({"entries": self.upload_session})

        async with Request(self.client_session.post, url, headers=headers, data=data) as resp:
            resp_data = await resp.json()
            self.upload_session = []  # empty the local upload session

            await logger.complete()

            if resp_data[".tag"] == "async_job_id":
                # check regularly for job completion
                return await self._upload_finish_check(resp_data["async_job_id"], check_interval=check_interval)
            elif resp_data[".tag"] == "complete":
                logger.info("Upload batch finished")
                return resp_data["entries"]
            else:
                err = await resp.text()
                raise DropboxAPIError(resp.status, f"Unknown upload_finish response: {err}")

    async def _upload_finish_check(self, job_id: str, check_interval: float = 5) -> list[dict[str, Any]]:
        """Checks on an upload_finish async job periodically.

        Should not be called directly, this is automatically called from upload_finish.
        See: https://www.dropbox.com/developers/documentation/http/documentation#files-upload_session-finish_batch-check

        Args:
            job_id: Job ID to check status of
            check_interval: How often in seconds to check status

        Returns:
            List of FileMetadata dicts containing metadata on each uploaded file
        """
        logger.debug(f"Batch not finished, checking every {check_interval} seconds")

        url = "https://api.dropboxapi.com/2/files/upload_session/finish_batch/check"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = json.dumps({"async_job_id": job_id})

        while True:
            await asyncio.sleep(check_interval)
            async with Request(self.client_session.post, url, headers=headers, data=data) as resp:
                resp_data = await resp.json()

                if resp_data[".tag"] == "complete":
                    logger.info("Upload batch finished")
                    return resp_data["entries"]
                elif resp_data[".tag"] == "in_progress":
                    logger.debug(f"Checking again in {check_interval} seconds")
                    continue

    async def upload_single(
        self, local_path: str, dropbox_path: str, args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Uploads a single file.

        This should only be used for small quantities of files.
        For larger quantities use upload_start and upload_finish.
        See: https://www.dropbox.com/developers/documentation/http/documentation#files-upload

        Args:
            local_path: Local path to upload from
            dropbox_path: Dropbox path to upload to
            args: Dictionary of arguments to pass to API

        Returns:
            FileMetadata of the uploaded file

        Raises:
            ValueError: If local_path does not exist
        """
        if args is None:
            args = {"mode": "add", "autorename": False, "mute": False}
        if not os.path.exists(local_path):
            raise ValueError(f"local_path {local_path} does not exist")
        args["path"] = dropbox_path

        logger.info(f"Uploading {os.path.basename(local_path)}")
        logger.debug(f"to {dropbox_path}")

        url = "https://content.dropboxapi.com/2/files/upload"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Dropbox-API-Arg": json.dumps(args),
            "Content-Type": "application/octet-stream",
        }

        async with aiofiles.open(local_path, "rb") as f:
            data = await f.read()
            async with Request(self.client_session.post, url, headers=headers, data=data) as resp:
                return await resp.json()

    async def create_shared_link(self, dropbox_path: str) -> str:
        """Create a shared link for a file in Dropbox.

        See: https://www.dropbox.com/developers/documentation/http/documentation#sharing-create_shared_link_with_settings

        Args:
            dropbox_path: Path of file on Dropbox to create shared link for

        Returns:
            Shared link for the given file.
            If shared link exists, returns existing one, otherwise creates new one.

        Raises:
            DropboxAPIError: If dropbox_path doesn't exist or unknown status returned
        """
        logger.info(f"Creating shared link for file {os.path.basename(dropbox_path)}")
        logger.debug(f"Full path is {dropbox_path}")

        url = "https://api.dropboxapi.com/2/sharing/create_shared_link_with_settings"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = json.dumps({"path": dropbox_path})

        # accept 409 status to check for existing shared link
        async with Request(
            self.client_session.post,
            url,
            headers=headers,
            data=data,
            ok_statuses=[200, 409],
        ) as resp:
            resp_data = await resp.json()

            if resp.status == 200:
                return resp_data["url"]
            if "shared_link_already_exists" in resp_data["error_summary"]:
                logger.warning(
                    f"Shared link already exists for {os.path.basename(dropbox_path)}, using existing link"
                )
                return resp_data["error"]["shared_link_already_exists"]["metadata"]["url"]
            elif "not_found" in resp_data["error_summary"]:
                raise DropboxAPIError(resp.status, f"Path {dropbox_path} does not exist")
            else:
                err = await resp.text()
                raise DropboxAPIError(resp.status, f"Unknown Dropbox error: {err}")

    async def get_shared_link_metadata(self, shared_link: str) -> dict[str, Any]:
        """Gets metadata for file/folder behind a shared link.

        See: https://www.dropbox.com/developers/documentation/http/documentation#sharing-get_shared_link_metadata

        Args:
            shared_link: Shared link pointing to file/folder to get metadata from

        Returns:
            FileMetadata or FolderMetadata for the file/folder
        """
        logger.info(f"Getting metadata from shared link {shared_link}")

        url = "https://api.dropboxapi.com/2/sharing/get_shared_link_metadata"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        data = json.dumps({"url": shared_link})

        async with Request(self.client_session.post, url, headers=headers, data=data) as resp:
            result = await resp.json()
            await logger.complete()
            return result

    async def __aenter__(self) -> AsyncDropboxAPI:
        await logger.complete()
        return self

    async def __aexit__(self, *excinfo: Any) -> None:
        await self.client_session.close()
        await logger.complete()


async def run_upload_to_dropbox(dbx: AsyncDropboxAPI, path_to_file: pathlib.PosixPath) -> dict[str, Any]:
    """Upload a file to Dropbox using an upload session.

    Args:
        dbx: AsyncDropboxAPI instance
        path_to_file: Path to file to upload

    Returns:
        Dict containing the commit information for the upload
    """
    # upload the new file to an upload session
    # this returns a "commit" dict, which will be passed to upload_finish later
    # the commit is saved in the AsyncDropboxAPI object already, so unless you need
    # information from it you can discard the return value
    result = await dbx.upload_start(path_to_file, f"/{pathlib.Path(path_to_file).name}")
    await logger.complete()
    return result


async def dropbox_upload(list_of_files_to_upload: list[str]) -> None:
    """Async upload function for dropbox.

    Call this to kick off a dbx.upload_start.

    Args:
        list_of_files_to_upload: List of file paths to upload
    """
    async with AsyncDropboxAPI(aiosettings.dropbox_cerebro_token.get_secret_value()) as dbx:  # pylint: disable=no-member
        await dbx.validate()

        coroutines = [run_upload_to_dropbox(dbx, _file) for _file in list_of_files_to_upload]
        for coro in asyncio.as_completed(coroutines):
            try:
                res = await coro
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
            else:
                logger.info(f"Processed {res}")

        # once everything is uploaded, finish the upload batch
        # this returns the metadata of all of the uploaded files
        await dbx.upload_finish()

        # print out some info
        logger.info("\nThe files we just uploaded are:")
        for meme in list_of_files_to_upload:
            logger.info(f"{meme}")

        await logger.complete()
