"""Asynchronous wrapper around gallery-dl."""
from __future__ import annotations

import asyncio

from collections.abc import AsyncIterator, Callable
from functools import partial
from typing import Any, Dict, Optional, TypeVar, cast, overload

import gallery_dl

from loguru import logger


T = TypeVar("T")
R = TypeVar("R")


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
    ) -> None:
        """Initialize AsyncGalleryDL.

        Args:
            config: Gallery-dl configuration dictionary
            loop: Optional asyncio event loop to use

        Example:
            >>> client = AsyncGalleryDL({"your": "config"})
        """
        self.config = config or {}
        self.loop = loop or asyncio.get_event_loop()

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
                gallery_dl.extractor.from_url,  # type: ignore[attr-defined] # pylint: disable=no-member
                url,
                self.config
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
