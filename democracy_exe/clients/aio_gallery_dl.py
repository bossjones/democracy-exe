"""Asynchronous wrapper around gallery-dl."""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import sys
import traceback

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional, TypeVar, Union, cast, overload

import aiofiles
import bpdb
import gallery_dl

from loguru import logger
from pydantic import BaseModel, EmailStr, Field

from democracy_exe.aio_settings import aiosettings
from democracy_exe.utils.file_functions import expand_path_str, tilda


T = TypeVar("T")
R = TypeVar("R")






from pydantic import ConfigDict, SecretStr


class HttpConfig(BaseModel):
    """Configuration for HTTP downloader settings."""
    model_config = ConfigDict(populate_by_name=True)

    adjust_extensions: bool = Field(True, alias="adjust-extensions")
    mtime: bool = True
    rate: int | None = None
    retries: int = 4
    timeout: float = 30.0
    verify: bool = True

class YtdlConfig(BaseModel):
    """Configuration for youtube-dl downloader settings."""
    model_config = ConfigDict(populate_by_name=True)

    format: str | None = None
    forward_cookies: bool = Field(False, alias="forward-cookies")
    logging: bool = True
    mtime: bool = True
    outtmpl: str | None = None
    rate: int | None = None
    retries: int = 4
    timeout: float = 30.0
    verify: bool = True

class DownloaderConfig(BaseModel):
    """Configuration for downloader settings."""
    model_config = ConfigDict(populate_by_name=True)

    filesize_min: int | None = Field(None, alias="filesize-min")
    filesize_max: int | None = Field(None, alias="filesize-max")
    part: bool = True
    part_directory: str | None = Field(None, alias="part-directory")
    http: HttpConfig
    ytdl: YtdlConfig

class OutputConfig(BaseModel):
    """Configuration for output settings."""
    model_config = ConfigDict(populate_by_name=True)

    mode: str = "auto"
    progress: bool = True
    shorten: bool = True
    log: str = "[{name}][{levelname}][{extractor.url}] {message}"
    logfile: str | None = None
    unsupportedfile: str | None = None

class DirectoryConfig(BaseModel):
    """Configuration for directory settings."""
    model_config = ConfigDict(populate_by_name=True)

    directory: list[str]

class InstagramConfig(BaseModel):
    """Configuration for Instagram extractor."""
    model_config = ConfigDict(populate_by_name=True)

    highlights: bool = False
    videos: bool = True
    include: str = "all"
    directory: list[str]
    stories: DirectoryConfig
    channel: DirectoryConfig
    tagged: DirectoryConfig
    reels: DirectoryConfig
    filename: str
    date_format: str = Field(alias="date-format")
    cookies: str | None = None
    username: SecretStr | None = None
    password: SecretStr | None = None
    sleep_request: float = Field(8.0, alias="sleep-request")

class RedditConfig(BaseModel):
    """Configuration for Reddit extractor."""
    model_config = ConfigDict(populate_by_name=True)

    client_id: SecretStr = Field(alias="client-id")
    user_agent: str = Field(alias="user-agent")
    browser: str
    refresh_token: SecretStr | None = Field(None, alias="refresh-token")
    comments: int = 0
    morecomments: bool = False
    date_min: int = Field(0, alias="date-min")
    date_max: int = Field(253402210800, alias="date-max")
    date_format: str = Field(alias="date-format")
    id_min: str | None = Field(None, alias="id-min")
    id_max: str | None = Field(None, alias="id-max")
    recursion: int = 0
    videos: bool = True
    parent_directory: bool = Field(True, alias="parent-directory")
    directory: list[str]
    filename: str

class TwitterConfig(BaseModel):
    """Configuration for Twitter extractor."""
    model_config = ConfigDict(populate_by_name=True)

    quoted: bool = True
    replies: bool = True
    retweets: bool = True
    twitpic: bool = False
    videos: bool = True
    cookies: str | None = None
    filename: str

class DeviantartConfig(BaseModel):
    """Configuration for DeviantArt extractor."""
    model_config = ConfigDict(populate_by_name=True)

    extra: bool = False
    flat: bool = True
    folders: bool = False
    journals: str = "html"
    mature: bool = True
    metadata: bool = False
    original: bool = True
    quality: int = 100
    wait_min: int = Field(0, alias="wait-min")

class PixivConfig(BaseModel):
    """Configuration for Pixiv extractor."""
    model_config = ConfigDict(populate_by_name=True)

    username: SecretStr | None = None
    password: SecretStr | None = None
    avatar: bool = False
    ugoira: bool = True

class ExtractorConfig(BaseModel):
    """Configuration for extractors."""
    model_config = ConfigDict(populate_by_name=True)

    base_directory: str = Field("./gallery-dl/", alias="base-directory")
    postprocessors: Any | None = None
    archive: str | None = None
    cookies: str | None = None
    cookies_update: bool = Field(True, alias="cookies-update")
    proxy: str | None = None
    skip: bool = True
    sleep: int = 0
    sleep_request: int = Field(0, alias="sleep-request")
    sleep_extractor: int = Field(0, alias="sleep-extractor")
    path_restrict: str = Field("auto", alias="path-restrict")
    path_replace: str = Field("_", alias="path-replace")
    path_remove: str = Field("\\u0000-\\u001f\\u007f", alias="path-remove")
    user_agent: str = Field(alias="user-agent")
    path_strip: str = Field("auto", alias="path-strip")
    path_extended: bool = Field(True, alias="path-extended")
    extension_map: dict[str, str] = Field(alias="extension-map")
    instagram: InstagramConfig
    reddit: RedditConfig
    twitter: TwitterConfig
    deviantart: DeviantartConfig
    pixiv: PixivConfig

class GalleryDLConfig(BaseModel):
    """Root configuration model for gallery-dl."""
    model_config = ConfigDict(populate_by_name=True)

    extractor: ExtractorConfig
    downloader: DownloaderConfig
    output: OutputConfig
    netrc: bool = False


class AsyncGalleryDL:
    """Asynchronous wrapper around gallery-dl.

    This class provides an async interface to gallery-dl operations,
    running them in a thread pool to avoid blocking the event loop.

    Attributes:
        config: Gallery-dl configuration dictionary
        loop: Optional asyncio event loop

    Example:
        >>> async with AsyncGalleryDL() as client:
        ...     async for item in client.extract_from_url("https://example.com"):
        ...         print(item)
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        verbose: bool = False,
        write_info_json: bool = False,
        write_metadata: bool = False,
        no_mtime: bool = False,
        config_file: str | None = None,
    ) -> None:
        """Initialize AsyncGalleryDL.

        Args:
            config: Gallery-dl configuration dictionary
            loop: Optional asyncio event loop to use
            verbose: Enable verbose output
            write_info_json: Write info JSON files
            write_metadata: Write metadata files
            no_mtime: Don't set file modification times
            config_file: Path to gallery-dl config file (default: ~/.gallery-dl.conf)

        Example:
            >>> client = AsyncGalleryDL({"your": "config"})
        """
        self.config = config or {}
        logger.debug(f"Using self.config: {self.config}")

        if verbose:
            self.config["verbosity"] = 2
        if write_info_json:
            self.config["write-info-json"] = True
        if write_metadata:
            self.config["write-metadata"] = True
        if no_mtime:
            self.config["no-mtime"] = True
        self.loop = loop or asyncio.get_event_loop()

        # Load config file if specified
        if config_file:
            self.config_file = expand_path_str(config_file)
        else:
            self.config_file = expand_path_str("~/.gallery-dl.conf")
        logger.debug(f"Using self.config_file: {self.config_file}")


    @overload
    async def _run_in_executor(
        self, func: Callable[..., T], *args: Any, **kwargs: Any
    ) -> T:
        ...

    @overload
    async def _run_in_executor(
        self, func: Callable[..., R], *args: Any, **kwargs: Any
    ) -> R:
        ...

    async def _run_in_executor(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Run a function in the default executor.

        Args:
            func: Function to run
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function

        Example:
            >>> result = await client._run_in_executor(some_func, arg1, kwarg1="value")
        """
        partial_func = partial(func, *args, **kwargs)
        return await self.loop.run_in_executor(None, partial_func)

    async def extract_from_url(self, url: str) -> AsyncIterator[dict[str, Any]]:
        """Extract items from a URL asynchronously.

        Args:
            url: URL to extract from

        Yields:
            Extracted items from gallery-dl

        Raises:
            ValueError: If extraction fails
            RuntimeError: If gallery-dl encounters an error

        Example:
            >>> async for item in client.extract_from_url("https://example.com"):
            ...     print(item["title"])
        """
        try:
            extractor = await self._run_in_executor(
                gallery_dl.extractor.find,  # type: ignore[attr-defined] # pylint: disable=no-member
                url
            )

            # Create async iterator from sync iterator
            for item in extractor:
                yield cast(dict[str, Any], item)
                # Give control back to event loop
                await asyncio.sleep(0)

        except Exception as e:
            logger.error("Error in gallery-dl extraction")
            raise

    async def download(
        self,
        url: str,
        **options: Any
    ) -> AsyncIterator[dict[str, Any]]:
        """Download content from URL asynchronously.

        Args:
            url: URL to download from
            **options: Additional options to pass to gallery-dl

        Yields:
            Download progress and results

        Raises:
            ValueError: If download fails
            RuntimeError: If gallery-dl encounters an error

        Example:
            >>> async for status in client.download("https://example.com"):
            ...     print(status["progress"])
        """
        try:
            job = await self._run_in_executor(
                gallery_dl.job.DownloadJob,  # type: ignore[attr-defined]
                url,
                options
            )

            # Run download in executor and yield results
            for item in job.run():
                yield cast(dict[str, Any], item)
                # Give control back to event loop
                await asyncio.sleep(0)

        except Exception as e:
            print(f"{e}")
            exc_type, exc_value, exc_traceback = sys.exc_info()
            print(f"Error Class: {e.__class__}")
            output = f"[UNEXPECTED] {type(e).__name__}: {e}"
            print(output)
            print(f"exc_type: {exc_type}")
            print(f"exc_value: {exc_value}")
            traceback.print_tb(exc_traceback)
            await logger.complete()
            if aiosettings.dev_mode:
                bpdb.pm()

            logger.error("Error in gallery-dl download")
            raise

    @classmethod
    async def extract_metadata(
        cls,
        url: str,
        config: dict[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        """Extract metadata from URL asynchronously.

        This is a convenience class method that creates a temporary instance
        for metadata extraction.

        Args:
            url: URL to extract metadata from
            config: Optional gallery-dl configuration

        Yields:
            Metadata items from the URL

        Raises:
            ValueError: If metadata extraction fails
            RuntimeError: If gallery-dl encounters an error

        Example:
            >>> async for metadata in AsyncGalleryDL.extract_metadata("https://example.com"):
            ...     print(metadata["title"])
        """
        async with cls(config=config) as client:
            async for item in client.extract_from_url(url):
                yield item

    async def __aenter__(self) -> AsyncGalleryDL:
        """Enter async context.

        Returns:
            Self instance

        Example:
            >>> async with AsyncGalleryDL() as client:
            ...     # Use client here
            ...     pass
        """
        # Load config file if it exists
        if os.path.exists(self.config_file):
            try:
                async with aiofiles.open(self.config_file, encoding="utf-8") as f:
                    config_data = json.loads(await f.read())
                    self.config.update(config_data)
                logger.debug(f"Loaded gallery-dl config from {self.config_file}")
            except Exception as e:
                logger.error(f"Error loading gallery-dl config: {e}")

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred

        Example:
            >>> async with AsyncGalleryDL() as client:
            ...     # Context is automatically cleaned up after this block
            ...     pass
        """
        # Cleanup if needed
        pass


if __name__ == "__main__":
    import asyncio

    import rich

    from langsmith import tracing_context
    from loguru import logger

    async def main() -> None:
        """Run the AsyncGalleryDL tool asynchronously."""
        url = "https://x.com/Eminitybaba_/status/1868256259251863704"
        with tracing_context(enabled=False):

            # Test download
            try:
                # Test extraction with command line options
                client = AsyncGalleryDL(verbose=True, write_info_json=True, write_metadata=True, no_mtime=True)
                items = []
                async for item in client.extract_from_url(url):
                    items.append(item)
                rich.print(f"items: {items}")

            except Exception as ex:
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


    asyncio.run(main())
