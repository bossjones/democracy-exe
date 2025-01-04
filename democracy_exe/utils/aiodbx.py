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
import uuid

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from typing import Any, Dict, List, Optional, TypeVar, Union, cast

import aiofiles
import aiohttp
import bpdb

from loguru import logger
from pydantic import SecretStr

from democracy_exe.aio_settings import aiosettings


async def aio_path_exists(path: str | pathlib.Path) -> bool:
    """Async wrapper for path existence check.

    Args:
        path: Path to check

    Returns:
        bool: True if path exists
    """
    return await asyncio.to_thread(os.path.exists, path)

async def aio_path_basename(path: str | pathlib.Path) -> str:
    """Async wrapper for path basename.

    Args:
        path: Path to get basename from

    Returns:
        str: Basename of path
    """
    return await asyncio.to_thread(os.path.basename, path)

async def aio_path_getsize(path: str | pathlib.Path) -> int:
    """Async wrapper for getting file size.

    Args:
        path: Path to get size of

    Returns:
        int: Size of file in bytes
    """
    return await asyncio.to_thread(os.path.getsize, path)


class DropboxAPIError(Exception):
    """Exception raised for Dropbox API errors.

    Args:
        status: HTTP status code
        message: Error message
    """

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        super().__init__(message)


class Request:
    """Wrapper for HTTP requests with retry logic.

    Args:
        request_func: Async request function to call
        url: Request URL
        headers: Request headers
        data: Request data
        retry_count: Number of retries
    """

    def __init__(
        self,
        request_func: Callable[..., Awaitable[aiohttp.ClientResponse]],
        url: str,
        headers: dict[str, str],
        data: str | bytes | dict[str, Any] | None = None,
        retry_count: int = 5
    ) -> None:
        self.request_func = request_func
        self.url = url
        self.headers = headers
        self.data = data
        self.retry_count = retry_count
        self.response: aiohttp.ClientResponse | None = None

    async def _do_request(self) -> aiohttp.ClientResponse:
        """Execute the request with retry logic.

        Returns:
            aiohttp.ClientResponse: The response from the request

        Raises:
            DropboxAPIError: If request fails after retries
        """
        last_exception = None
        for attempt in range(self.retry_count):
            try:
                response = await self.request_func(
                    self.url,
                    headers=self.headers,
                    data=self.data
                )
                if response.status == 429:  # Rate limit
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                if response.status >= 400:
                    error_data = await response.text()
                    try:
                        error_json = json.loads(error_data)
                        error_message = error_json.get("error_summary", error_data)
                    except json.JSONDecodeError:
                        error_message = error_data
                    raise DropboxAPIError(response.status, str(error_message))
                return response
            except DropboxAPIError:
                raise
            except Exception as e:
                last_exception = e
                if attempt < self.retry_count - 1:
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                break

        raise DropboxAPIError(
            500,
            f"Request failed after {self.retry_count} retries: {last_exception!s}"
        )

    def __await__(self) -> Generator[Any, None, aiohttp.ClientResponse]:
        """Make the class awaitable.

        Returns:
            Generator yielding the response
        """
        return self._do_request().__await__()

    async def __aenter__(self) -> aiohttp.ClientResponse:
        """Enter async context manager.

        Returns:
            aiohttp.ClientResponse: The response from the request
        """
        self.response = await self._do_request()
        return self.response

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        if self.response:
            await self.response.release()


async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    **kwargs: Any
) -> Any:
    """Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        **kwargs: Keyword arguments for the function

    Returns:
        Result from the function

    Raises:
        Exception: The last exception encountered after all retries
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                raise

            logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e!s}")
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    raise last_exception if last_exception else RuntimeError("Unexpected retry failure")

class SafeFileHandler:
    """Safe file handler with proper error handling and cleanup.

    Args:
        path: Path to the file
        mode: File mode
        cleanup_on_error: Whether to delete file on error in write mode
        **kwargs: Additional arguments for aiofiles.open
    """

    def __init__(
        self,
        path: str | pathlib.Path,
        mode: str = "r",
        cleanup_on_error: bool = True,
        **kwargs: Any
    ) -> None:
        self.path = path
        self.mode = mode
        self.cleanup_on_error = cleanup_on_error
        self.kwargs = kwargs
        self.file = None

    async def __aenter__(self) -> Any:
        """Enter the async context manager.

        Returns:
            The opened file object

        Raises:
            OSError: If file operations fail
        """
        try:
            self.file = await aiofiles.open(self.path, self.mode, **self.kwargs)
            return self.file
        except Exception as e:
            if self.cleanup_on_error and "w" in self.mode:
                try:
                    await asyncio.to_thread(os.remove, self.path)
                except:
                    pass
            raise OSError(f"File operation failed: {e!s}") from e

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager.

        Ensures proper cleanup of resources.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        if self.file:
            try:
                await self.file.close()
            except:
                pass

            if exc_type and self.cleanup_on_error and "w" in self.mode:
                try:
                    await asyncio.to_thread(os.remove, self.path)
                except:
                    pass

def safe_aiofiles_open(
    path: str | pathlib.Path,
    mode: str = "r",
    cleanup_on_error: bool = True,
    **kwargs: Any
) -> SafeFileHandler:
    """Create a safe file handler with proper error handling and cleanup.

    Args:
        path: Path to the file
        mode: File mode
        cleanup_on_error: Whether to delete file on error in write mode
        **kwargs: Additional arguments for aiofiles.open

    Returns:
        SafeFileHandler: A context manager for safe file operations
    """
    return SafeFileHandler(path, mode, cleanup_on_error, **kwargs)


class AsyncDropboxAPI:
    """Async Dropbox API client.

    Args:
        access_token: Dropbox access token
        max_retries: Maximum number of retries for failed requests
        retry_delay: Delay between retries in seconds
    """

    def __init__(
        self,
        access_token: str | None | SecretStr = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> None:
        if access_token is None:
            logger.debug("Using aiosettings.dropbox_cerebro_token")
            # self.access_token_secret: SecretStr = cast(SecretStr, aiosettings.dropbox_cerebro_token)
            self.access_token: SecretStr | str = aiosettings.dropbox_cerebro_token.get_secret_value()  # pylint: disable=no-member
        else:
            if isinstance(access_token, str):
                self.access_token: str | SecretStr = SecretStr(access_token)
            else:
                self.access_token: SecretStr | str = access_token

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client_session: aiohttp.ClientSession | None = None
        self.upload_session: list[dict[str, Any]] = []
        self._request_semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        self._upload_session_lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._closed = False

    async def _cleanup(self) -> None:
        """Clean up resources and close connections."""
        if self._closed:
            return

        async with self._cleanup_lock:
            if self._closed:
                return

            try:
                # Clear upload session
                async with self._upload_session_lock:
                    self.upload_session.clear()

                # Close client session
                if self.client_session and not self.client_session.closed:
                    await self.client_session.close()

                self._closed = True
                logger.debug("Cleanup completed successfully")

            except Exception as e:
                logger.error(f"Error during cleanup: {e!s}")
                raise DropboxAPIError(500, f"Cleanup failed: {e!s}")

    async def __aenter__(self) -> AsyncDropboxAPI:
        """Enter async context manager.

        Returns:
            Self instance
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context manager.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        await self._cleanup()

    async def _do_request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        data: str | bytes | dict[str, Any] | None = None,
        json_data: dict[str, Any] | None = None,
        retry_count: int = 0
    ) -> Any:
        """Make an HTTP request to the Dropbox API.

        Args:
            method: HTTP method
            url: Request URL
            headers: Optional request headers
            data: Optional request data
            json_data: Optional JSON data
            retry_count: Current retry attempt

        Returns:
            Response data as JSON

        Raises:
            DropboxAPIError: If request fails after retries
        """
        if not self.client_session:
            logger.debug("Creating new client session")
            self.client_session = aiohttp.ClientSession()

        headers = headers or {}
        if "Authorization" not in headers:
            logger.debug("Adding authorization header")
            headers["Authorization"] = f"Bearer {aiosettings.dropbox_cerebro_token.get_secret_value()}" # pylint: disable=no-member

        logger.debug(f"Making {method} request to {url} (retry {retry_count}/{self.max_retries})")
        logger.debug(f"Request headers: {headers}")
        if data:
            logger.debug(f"Request data length: {len(data) if isinstance(data, (str, bytes)) else len(str(data))} bytes")
        if json_data:
            logger.debug(f"Request JSON data: {json_data}")

        try:
            async with self._request_semaphore:
                logger.debug("Acquired request semaphore")
                async with self.client_session.request(
                    method,
                    url,
                    headers=headers,
                    data=data,
                    json=json_data
                ) as response:
                    logger.debug(f"Response status: {response.status}")
                    logger.debug(f"Response headers: {response.headers}")

                    if response.status == 429:  # Rate limit
                        if retry_count < self.max_retries:
                            delay = self.retry_delay * (2 ** retry_count)
                            logger.debug(f"Rate limited, retrying in {delay} seconds")
                            await asyncio.sleep(delay)
                            return await self._do_request(
                                method, url, headers, data,
                                json_data, retry_count + 1
                            )
                        logger.error("Rate limit exceeded after max retries")
                        raise DropboxAPIError(429, "Rate limit exceeded")

                    if not response.ok:
                        error_data = await response.text()
                        logger.error(f"Request failed with status {response.status}: {error_data}")
                        raise DropboxAPIError(
                            response.status,
                            f"Request failed: {response.status} - {error_data}"
                        )

                    if response.content_type == "application/json":
                        response_data = await response.json()
                        logger.debug(f"Received JSON response: {response_data}")
                        return response_data
                    response_text = await response.text()
                    logger.debug(f"Received text response length: {len(response_text)} bytes")
                    return response_text

        except aiohttp.ClientError as e:
            logger.error(f"Client error occurred: {e!s}")
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** retry_count)
                logger.debug(f"Retrying request in {delay} seconds")
                await asyncio.sleep(delay)
                return await self._do_request(
                    method, url, headers, data,
                    json_data, retry_count + 1
                )
            raise DropboxAPIError(500, f"Request failed: {e!s}")

        except Exception as e:
            logger.error(f"Unexpected error occurred: {e!s}")
            logger.error(f"Traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}")
            raise DropboxAPIError(500, f"Unexpected error: {e!s}")

    async def validate(self) -> bool:
        """Validate the API token.

        Returns:
            bool: True if token is valid

        Raises:
            DropboxAPIError: If validation fails
        """
        logger.info("Validating Dropbox API token...")
        nonce = str(uuid.uuid4())
        url = "https://api.dropboxapi.com/2/check/user"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        data = json.dumps({"query": nonce})

        try:
            response = await self._do_request("POST", url, headers=headers, data=data)
            if response["result"] != nonce:
                logger.error("API token validation failed: Invalid response")
                raise DropboxAPIError(401, "Invalid API token")
            logger.info("API token validated successfully")
            await logger.complete()
            return True

        except Exception as ex:
            import bpdb
            import rich
            logger.error(f"API token validation failed: {ex}")
            print(f"{ex}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f"Error Class: {ex.__class__}")
            output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
            print(output)
            print(f"exc_type: {exc_type}")
            print(f"exc_value: {exc_value}")
            traceback.print_tb(exc_traceback)
            await logger.complete()
            rich.print(f"aiosettings.dev_mode: {aiosettings.dev_mode}")
            if aiosettings.dev_mode:
                bpdb.pm()

            await logger.complete()
            raise DropboxAPIError(401, f"Token validation failed: {ex}")

        # except Exception as e:
        #     logger.error(f"API token validation failed: {e!s}")
        #     await logger.complete()
        #     raise DropboxAPIError(401, f"Token validation failed: {e!s}")

    async def download_file(self, dropbox_path: str, local_path: str | None = None) -> str:
        """Download a file from Dropbox.

        Args:
            dropbox_path: Path to file in Dropbox
            local_path: Optional local path to save file to

        Returns:
            str: Path to downloaded file

        Raises:
            DropboxAPIError: If download fails
            OSError: If file operations fail
        """
        if not local_path:
            local_path = os.path.basename(dropbox_path)

        logger.info(f"Downloading file from Dropbox: {dropbox_path}")
        logger.debug(f"Saving to local path: {local_path}")

        url = "https://content.dropboxapi.com/2/files/download"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Dropbox-API-Arg": json.dumps({"path": dropbox_path}),
        }

        try:
            response = await self._do_request("POST", url, headers=headers)
            async with safe_aiofiles_open(local_path, "wb") as f:
                if isinstance(response, str):
                    await f.write(response.encode())
                else:
                    await f.write(response)
            logger.info(f"File downloaded successfully: {local_path}")
            await logger.complete()
            return local_path
        except Exception as e:
            logger.error(f"File download failed: {e!s}")
            await logger.complete()
            raise DropboxAPIError(500, f"Download failed: {e!s}")

    async def download_folder(self, dropbox_path: str, local_path: str | None = None) -> str:
        """Download a folder from Dropbox as a zip file.

        Args:
            dropbox_path: Path to folder in Dropbox
            local_path: Optional local path to save zip file to

        Returns:
            str: Path to downloaded zip file

        Raises:
            DropboxAPIError: If download fails
            OSError: If file operations fail
        """
        if not local_path:
            local_path = os.path.basename(dropbox_path) + ".zip"

        logger.info(f"Downloading folder from Dropbox: {dropbox_path}")
        logger.debug(f"Saving to local path: {local_path}")

        url = "https://content.dropboxapi.com/2/files/download_zip"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Dropbox-API-Arg": json.dumps({"path": dropbox_path}),
        }

        try:
            response = await self._do_request("POST", url, headers=headers)
            async with safe_aiofiles_open(local_path, "wb") as f:
                if isinstance(response, str):
                    await f.write(response.encode())
                else:
                    await f.write(response)
            logger.info(f"Folder downloaded successfully: {local_path}")
            await logger.complete()
            return local_path
        except Exception as e:
            logger.error(f"Folder download failed: {e!s}")
            await logger.complete()
            raise DropboxAPIError(500, f"Download failed: {e!s}")

    async def download_shared_link(self, shared_link: str, local_path: str | None = None) -> str:
        """Download a file from a shared link.

        Args:
            shared_link: Shared link URL
            local_path: Optional local path to save file to

        Returns:
            str: Path to downloaded file

        Raises:
            DropboxAPIError: If download fails
            OSError: If file operations fail
        """
        if not local_path:
            local_path = str(uuid.uuid4())

        url = "https://content.dropboxapi.com/2/sharing/get_shared_link_file"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Dropbox-API-Arg": json.dumps({"url": shared_link}),
        }

        try:
            response = await self._do_request("POST", url, headers=headers)
            async with safe_aiofiles_open(local_path, "wb") as f:
                if isinstance(response, str):
                    await f.write(response.encode())
                else:
                    await f.write(response)
            await logger.complete()
            return local_path
        except Exception as e:
            await logger.complete()
            raise DropboxAPIError(500, f"Download failed: {e!s}")

    async def upload_start(self, local_path: str, dropbox_path: str) -> dict[str, Any]:
        """Start an upload session.

        Args:
            local_path: Path to local file
            dropbox_path: Destination path in Dropbox

        Returns:
            Dict containing upload session information

        Raises:
            DropboxAPIError: If upload fails
            OSError: If file operations fail
        """
        logger.info(f"Starting upload session for file: {local_path}")
        logger.debug(f"Destination path: {dropbox_path}")

        url = "https://content.dropboxapi.com/2/files/upload_session/start"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Dropbox-API-Arg": json.dumps({"close": True}),
            "Content-Type": "application/octet-stream",
        }

        try:
            async with safe_aiofiles_open(local_path, "rb") as f:
                data = await f.read()
                response = await self._do_request("POST", url, headers=headers, data=data)
                logger.info("Upload session started successfully")
                await logger.complete()
                return response
        except Exception as e:
            logger.error(f"Failed to start upload session: {e!s}")
            await logger.complete()
            raise DropboxAPIError(500, f"Upload failed: {e!s}")

    async def upload_finish(self) -> dict[str, Any]:
        """Finish the upload session.

        Returns:
            Dict containing upload completion information

        Raises:
            DropboxAPIError: If upload fails
        """
        if not self.upload_session:
            logger.error("No active upload session to finish")
            await logger.complete()
            raise DropboxAPIError(400, "No active upload session")

        logger.info("Finishing upload session...")
        logger.debug(f"Number of files in session: {len(self.upload_session)}")

        url = "https://api.dropboxapi.com/2/files/upload_session/finish_batch"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        data = json.dumps({"entries": self.upload_session})

        try:
            response = await self._do_request("POST", url, headers=headers, data=data)
            session_id = response.get("async_job_id")
            if not session_id:
                logger.error("No async job ID received in response")
                await logger.complete()
                raise DropboxAPIError(500, "No async job ID in response")

            logger.info("Waiting for upload completion...")
            # Poll for completion
            url = "https://api.dropboxapi.com/2/files/upload_session/finish_batch/check"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }
            data = json.dumps({"async_job_id": session_id})

            while True:
                response = await self._do_request("POST", url, headers=headers, data=data)
                if response[".tag"] != "in_progress":
                    break
                logger.debug("Upload still in progress, checking again...")
                await asyncio.sleep(1)

            logger.info("Upload session completed successfully")
            await logger.complete()
            return response
        except Exception as e:
            logger.error(f"Upload session failed: {e!s}")
            await logger.complete()
            raise DropboxAPIError(500, f"Upload finish failed: {e!s}")
        finally:
            self.upload_session = []

    async def upload_single(
        self,
        local_path: str,
        dropbox_path: str,
        mode: str = "add",
        autorename: bool = False,
        mute: bool = False
    ) -> dict[str, Any]:
        """Upload a single file to Dropbox.

        Args:
            local_path: Path to local file
            dropbox_path: Destination path in Dropbox
            mode: Upload mode (add, overwrite)
            autorename: Whether to rename file if it exists
            mute: Whether to mute notifications

        Returns:
            Dict containing upload metadata

        Raises:
            DropboxAPIError: If upload fails
            OSError: If file operations fail
        """
        logger.info(f"Uploading file to Dropbox: {local_path}")
        logger.debug(f"Destination path: {dropbox_path} (mode: {mode})")

        args = {
            "path": dropbox_path,
            "mode": mode,
            "autorename": autorename,
            "mute": mute,
        }

        url = "https://content.dropboxapi.com/2/files/upload"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Dropbox-API-Arg": json.dumps(args),
            "Content-Type": "application/octet-stream",
        }

        try:
            async with safe_aiofiles_open(local_path, "rb") as f:
                data = await f.read()
                response = await self._do_request("POST", url, headers=headers, data=data)
                logger.info(f"File uploaded successfully: {dropbox_path}")
                await logger.complete()
                return response
        except Exception as e:
            logger.error(f"File upload failed: {e!s}")
            await logger.complete()
            raise DropboxAPIError(500, f"Upload failed: {e!s}")

    async def create_shared_link(self, dropbox_path: str) -> str:
        """Create a shared link for a file.

        Args:
            dropbox_path: Path to file in Dropbox

        Returns:
            str: Shared link URL

        Raises:
            DropboxAPIError: If request fails
        """
        logger.info(f"Creating shared link for: {dropbox_path}")

        url = "https://api.dropboxapi.com/2/sharing/create_shared_link_with_settings"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        data = json.dumps({"path": dropbox_path})

        try:
            response = await self._do_request("POST", url, headers=headers, data=data)
            logger.info("Shared link created successfully")
            await logger.complete()
            return response["url"]
        except DropboxAPIError as e:
            if e.status == 409:  # Link already exists
                logger.info("Shared link already exists, retrieving existing link")
                url = "https://api.dropboxapi.com/2/sharing/get_shared_link_metadata"
                response = await self._do_request("POST", url, headers=headers, data=data)
                await logger.complete()
                return response["url"]
            logger.error(f"Failed to create shared link: {e!s}")
            await logger.complete()
            raise

    async def get_shared_link_metadata(self, shared_link: str) -> dict[str, Any]:
        """Get metadata for a shared link.

        Args:
            shared_link: Path to file in Dropbox

        Returns:
            Dict containing shared link metadata

        Raises:
            DropboxAPIError: If request fails
        """
        logger.info(f"Getting metadata for shared link: {shared_link}")

        url = "https://api.dropboxapi.com/2/sharing/create_shared_link_with_settings"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json",
        }
        data = json.dumps({"path": shared_link})

        try:
            response = await self._do_request("POST", url, headers=headers, data=data)
            logger.info("Retrieved shared link metadata successfully")
            await logger.complete()
            return response
        except DropboxAPIError as e:
            if e.status == 409:  # Link already exists
                logger.info("Using existing shared link metadata")
                url = "https://api.dropboxapi.com/2/sharing/get_shared_link_metadata"
                response = await self._do_request("POST", url, headers=headers, data=data)
                await logger.complete()
                return response
            logger.error(f"Failed to get shared link metadata: {e!s}")
            await logger.complete()
            raise

    async def get_connection_metrics(self) -> dict[str, Any]:
        """Get current connection pooling metrics.

        Returns:
            Dictionary containing connection metrics:
                - active_connections: Number of currently active connections
                - acquired_connections: Number of connections currently in use
                - connection_limit: Maximum allowed connections
                - connection_timeouts: Number of connection timeouts
        """
        logger.debug("Getting connection pooling metrics")

        if not self.client_session or self.client_session.closed:
            logger.debug("No active client session")
            return {
                "active_connections": 0,
                "acquired_connections": 0,
                "connection_limit": 0,
                "connection_timeouts": 0
            }

        connector = self.client_session.connector
        if not connector:
            logger.debug("No active connector")
            return {
                "active_connections": 0,
                "acquired_connections": 0,
                "connection_limit": 0,
                "connection_timeouts": 0
            }

        metrics = {
            "active_connections": len(connector._conns),  # type: ignore
            "acquired_connections": len(connector._acquired),  # type: ignore
            "connection_limit": connector._limit,  # type: ignore
            "connection_timeouts": getattr(connector, "_timeouts_count", 0)  # type: ignore
        }
        logger.debug(f"Connection metrics: {metrics}")
        return metrics

    async def _recover_upload_session(self) -> None:
        """Attempt to recover a failed upload session.

        This method tries to salvage any valid uploads from a failed session
        and cleans up any incomplete uploads.

        Raises:
            DropboxAPIError: If session recovery fails
        """
        if not self.upload_session:
            return

        async with self._upload_session_lock:
            try:
                # Get list of valid session IDs
                valid_sessions = []
                for entry in self.upload_session:
                    try:
                        session_id = entry["cursor"]["session_id"]
                        # Try to verify session is still valid
                        url = "https://api.dropboxapi.com/2/files/upload_session/finish_batch/check"
                        headers = {
                            "Authorization": f"Bearer {self.access_token}",
                            "Content-Type": "application/json",
                        }
                        data = json.dumps({"async_job_id": session_id})
                        response = await self._do_request("POST", url, headers=headers, data=data)
                        if response[".tag"] != "in_progress":
                            continue
                        valid_sessions.append(entry)
                    except:
                        continue

                # Update upload session with only valid entries
                self.upload_session = valid_sessions

                if len(valid_sessions) > 0:
                    logger.info(f"Recovered {len(valid_sessions)} valid upload sessions")
                else:
                    logger.warning("No valid upload sessions could be recovered")

            except Exception as e:
                logger.error(f"Failed to recover upload session: {e!s}")
                # Clear the session if recovery fails
                self.upload_session.clear()
                raise DropboxAPIError(500, f"Failed to recover upload session: {e!s}")

    async def dropbox_upload(
        self,
        file_path: str | pathlib.Path,
        dropbox_path: str,
        chunk_size: int = 4 * 1024 * 1024,  # 4MB chunks
        overwrite: bool = True,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> str:
        """Upload a file to Dropbox.

        Args:
            file_path: Local path to file
            dropbox_path: Dropbox destination path
            chunk_size: Size of upload chunks in bytes
            overwrite: Whether to overwrite existing file
            progress_callback: Optional callback for upload progress

        Returns:
            str: Shared link to uploaded file

        Raises:
            DropboxAPIError: If upload fails
            OSError: If file operations fail
        """
        logger.info(f"Starting chunked upload to Dropbox: {file_path}")
        logger.debug(f"Destination path: {dropbox_path} (chunk size: {chunk_size/1024/1024:.1f}MB)")

        try:
            async with safe_aiofiles_open(file_path, "rb") as f:
                file_size = await asyncio.to_thread(os.path.getsize, file_path)
                uploaded = 0

                # Start upload session
                logger.info("Initializing upload session...")
                upload_session_start_result = await self._do_request(
                    "POST",
                    "https://content.dropboxapi.com/2/files/upload_session/start",
                    headers={"Content-Type": "application/octet-stream"},
                    data=b"",
                )
                session_id = upload_session_start_result["session_id"]
                logger.debug(f"Upload session initialized with ID: {session_id}")

                # Upload file chunks
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break

                    if uploaded + len(chunk) < file_size:
                        # Append to upload session
                        logger.debug(f"Uploading chunk: {uploaded/1024/1024:.1f}MB / {file_size/1024/1024:.1f}MB")
                        await self._do_request(
                            "POST",
                            "https://content.dropboxapi.com/2/files/upload_session/append_v2",
                            headers={
                                "Content-Type": "application/octet-stream",
                                "Dropbox-API-Arg": json.dumps({
                                    "cursor": {
                                        "session_id": session_id,
                                        "offset": uploaded
                                    }
                                })
                            },
                            data=chunk,
                        )
                    else:
                        # Finish upload session
                        logger.info("Finalizing upload...")
                        mode = "overwrite" if overwrite else "add"
                        finish_args = {
                            "cursor": {
                                "session_id": session_id,
                                "offset": uploaded
                            },
                            "commit": {
                                "path": dropbox_path,
                                "mode": mode
                            }
                        }

                        await self._do_request(
                            "POST",
                            "https://content.dropboxapi.com/2/files/upload_session/finish",
                            headers={
                                "Content-Type": "application/octet-stream",
                                "Dropbox-API-Arg": json.dumps(finish_args)
                            },
                            data=chunk,
                        )

                    uploaded += len(chunk)
                    if progress_callback:
                        await progress_callback(uploaded, file_size)

                # Get shared link
                logger.info("Upload completed, creating shared link...")
                shared_link = await self.get_shared_link_metadata(dropbox_path)
                logger.info("File uploaded and shared successfully")
                await logger.complete()
                return shared_link["url"]

        except DropboxAPIError:
            logger.error("Upload failed due to Dropbox API error")
            raise
        except Exception as e:
            error_msg = f"Upload failed: {e!s}"
            logger.error(error_msg)
            await logger.complete()
            raise DropboxAPIError(500, error_msg) from e
