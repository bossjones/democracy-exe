"""Extension utility functions.

This module provides utilities for managing Discord bot extensions/cogs.
"""
from __future__ import annotations

import os
import pathlib

from collections.abc import AsyncIterator, Iterable
from typing import TYPE_CHECKING, Any, List

import aiofiles
import pysnooper

# from loguru import logger
import structlog

from discord.ext import commands


logger = structlog.get_logger(__name__)

from democracy_exe.aio_settings import aiosettings


HERE = os.path.dirname(os.path.dirname(__file__))
print(f"HERE: {HERE}")
print(f"HERE: {HERE}")
print(f"HERE: {HERE}")
print(f"HERE: {HERE}")

if TYPE_CHECKING:
    from democracy_exe.chatbot.core.bot import DemocracyBot

def extensions() -> Iterable[str]:
    """Yield extension module paths.

    This function searches for Python files in the 'cogs' directory relative to the current file's directory.
    It constructs the module path for each file and yields it.

    Yields:
        The module path for each Python file in the 'cogs' directory

    Raises:
        FileNotFoundError: If cogs directory doesn't exist
    """
    cogs_dir = pathlib.Path(HERE) / "cogs"
    logger.error(f"Cogs directory: {cogs_dir}")
    if not cogs_dir.exists():
        raise FileNotFoundError(f"Cogs directory not found: {cogs_dir}")

    for file in cogs_dir.rglob("*.py"):
        logger.error(f"file: {file}")
        logger.error(f"file.name: {file.name}")
        logger.error(f"aiosettings.extension_allowlist: {aiosettings.extension_allowlist}")
        module_name = file.name.replace(".py", "")
        logger.error(f"module_name: {module_name}")
        logger.error(f"file.name != '__init__.py': {file.name != '__init__.py'}")
        logger.error(f"module_name in aiosettings.extension_allowlist: {module_name in aiosettings.extension_allowlist}")
        is_allowed = any(module_name in item for item in aiosettings.extension_allowlist)
        logger.error(f"is_allowed: {is_allowed}")
        if file.name != "__init__.py" and is_allowed:
            # Get path relative to the module root
            base_module_dir = pathlib.Path(HERE).parent.parent

            # relative_path = file.relative_to(pathlib.Path(HERE))
            relative_path = file.relative_to(base_module_dir)
            logger.error(f"Relative path: {relative_path}")
            extension_path = str(relative_path)[:-3].replace(os.sep, ".")
            logger.error(f"Extension path: {extension_path}")
            logger.debug(f"Found extension file: {file}")
            logger.debug(f"Converting to module path: {extension_path}")
            yield extension_path


class AsyncExtensionIterator:
    """Async iterator for discovering extensions."""

    def __init__(self) -> None:
        """Initialize the iterator."""
        self.cogs_path = pathlib.Path(HERE) / "cogs"
        logger.error(f"Cogs path: {self.cogs_path}")
        self.files = None
        self.current_index = 0

    def __aiter__(self) -> AsyncExtensionIterator:
        """Return self as the iterator.

        Returns:
            Self as the async iterator
        """
        return self

    # @pysnooper(thread_info=True, max_variable_length=None, depth=10)
    async def __anext__(self) -> str:
        """Get the next extension path.

        Returns:
            The next extension path

        Raises:
            StopAsyncIteration: When no more extensions are available
            FileNotFoundError: If cogs directory doesn't exist
        """
        if not self.cogs_path.exists():
            logger.error(f"Cogs directory not found: {self.cogs_path}")
            raise FileNotFoundError(f"Cogs directory not found: {self.cogs_path}")

        if self.files is None:
            # Initialize the file list on first iteration
            self.files = list(self.cogs_path.rglob("*.py"))
            logger.debug(f"Found files: {self.files}")
            logger.debug("Successfully initialized async file search")

        while self.current_index < len(self.files):
            file = self.files[self.current_index]
            self.current_index += 1

            # Skip __init__.py files
            logger.error(f"file: {file}")
            logger.error(f"file.name: {file.name}")
            logger.error(f"aiosettings.extension_allowlist: {aiosettings.extension_allowlist}")
            module_name = file.name.replace(".py", "")
            logger.error(f"module_name: {module_name}")
            logger.error(f"file.name != '__init__.py': {file.name != '__init__.py'}")
            logger.error(f"module_name in aiosettings.extension_allowlist: {module_name in aiosettings.extension_allowlist}")
            is_allowed = any(module_name in item for item in aiosettings.extension_allowlist)
            logger.error(f"is_allowed: {is_allowed}")

            # if file is __init__.py or not in allowlist, skip
            if file.name == "__init__.py" or not is_allowed:
                continue

            try:
                # Verify file exists and is readable
                async with aiofiles.open(file) as f:
                    # Just check if we can open it
                    await f.read(1)



                # # For testing:
                # >>> HERE = "/Users/malcolm/dev/bossjones/democracy-exe/democracy_exe/chatbot"
                # >>> cogs_path = pathlib.Path("/Users/malcolm/dev/bossjones/democracy-exe/democracy_exe/chatbot/cogs")
                # >>> file = pathlib.Path("/Users/malcolm/dev/bossjones/democracy-exe/democracy_exe/chatbot/cogs/twitter.py")
                # >>> base_module_dir = pathlib.Path(HERE).parent.parent
                # >>> relative_path = file.relative_to(base_module_dir)
                # >>> extension_path = str(relative_path).replace(os.sep, ".")[:-3]
                # >>> extension_path
                # 'democracy_exe.chatbot.cogs.twitter'
                # >>> print(f"HERE: {HERE}")
                # HERE: /Users/malcolm/dev/bossjones/democracy-exe/democracy_exe/chatbot
                # >>> print(f"cogs_path: {cogs_path}")
                # cogs_path: /Users/malcolm/dev/bossjones/democracy-exe/democracy_exe/chatbot/cogs
                # >>> print(f"file: {file}")
                # file: /Users/malcolm/dev/bossjones/democracy-exe/democracy_exe/chatbot/cogs/twitter.py
                # >>> print(f"base_module_dir: {base_module_dir}")
                # base_module_dir: /Users/malcolm/dev/bossjones/democracy-exe
                # >>> print(f"relative_path: {relative_path}")
                # relative_path: democracy_exe/chatbot/cogs/twitter.py
                # >>> print(f"extension_path: {extension_path}")
                # extension_path: democracy_exe.chatbot.cogs.twitter
                # >>>


                # Get path relative to the cogs directory
                base_module_dir = pathlib.Path(HERE).parent.parent

                logger.error(f"base_module_dir: {base_module_dir}")

                relative_path = file.relative_to(base_module_dir)
                logger.error(f"relative_path: {relative_path}")

                extension_path = str(relative_path).replace(os.sep, ".")[:-3]
                logger.error(f"extension_path: {extension_path}")
                logger.error(f"file: {file}")
                logger.error(f"HERE: {HERE}")
                logger.error(f"relative_path: {relative_path}")
                # import bpdb; bpdb.set_trace()


                logger.debug(f"Found extension file: {file}")
                logger.debug(f"Converting to module path: {extension_path}")
                # await logger.complete()
                return extension_path

            except OSError as e:
                logger.warning(f"Skipping inaccessible extension file {file}: {e}")
                continue

        logger.debug("Completed async extension discovery")
        # await logger.complete()
        raise StopAsyncIteration


async def aio_extensions() -> AsyncIterator[str]:
    """Yield extension module paths asynchronously.

    This function asynchronously searches for Python files in the 'cogs' directory
    relative to the current file's directory. It constructs the module path for each
    file and yields it. Uses aiofiles for asynchronous file operations.

    Yields:
        The module path for each Python file in the 'cogs' directory

    Raises:
        FileNotFoundError: If the cogs directory doesn't exist
        OSError: If there's an error accessing the cogs directory
    """
    try:
        async for extension in AsyncExtensionIterator():
            yield extension
    except Exception as e:
        logger.error(f"Error discovering extensions: {e}")
        logger.exception("Extension discovery failed")
        raise


async def load_extensions(bot: DemocracyBot, extension_list: list[str]) -> None:
    """Load a list of extensions into the bot.

    Args:
        bot: The Discord bot instance
        extension_list: List of extension module paths to load

    Raises:
        Exception: If loading an extension fails
    """
    for extension in extension_list:
        try:
            await bot.load_extension(extension)
            logger.info(f"Loaded extension: {extension}")
            # await logger.complete()
        except Exception as e:
            logger.error(f"Failed to load extension {extension}: {e}")
            # await logger.complete()
            raise


async def reload_extension(bot: DemocracyBot, extension: str) -> None:
    """Reload a specific extension.

    Args:
        bot: The Discord bot instance
        extension: Extension module path to reload

    Raises:
        Exception: If reloading the extension fails
    """
    try:
        await bot.reload_extension(extension)
        logger.info(f"Reloaded extension: {extension}")
        # await logger.complete()
    except Exception as e:
        logger.error(f"Failed to reload extension {extension}: {e}")
        # await logger.complete()
        raise


async def unload_extension(bot: DemocracyBot, extension: str) -> None:
    """Unload a specific extension.

    Args:
        bot: The Discord bot instance
        extension: Extension module path to unload

    Raises:
        Exception: If unloading the extension fails
    """
    try:
        await bot.unload_extension(extension)
        logger.info(f"Unloaded extension: {extension}")
        # await logger.complete()
    except Exception as e:
        logger.error(f"Failed to unload extension {extension}: {e}")
        # await logger.complete()
        raise
