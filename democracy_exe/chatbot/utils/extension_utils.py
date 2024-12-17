"""Extension management utilities for Discord bot.

This module handles loading and managing Discord bot extensions/cogs.
"""
from __future__ import annotations

import os
import pathlib

from collections.abc import AsyncIterator, Iterable
from typing import Any, List

import aiofiles

from loguru import logger


HERE = os.path.dirname(__file__)


def extensions() -> Iterable[str]:
    """Yield extension module paths synchronously.

    This function searches for Python files in the 'cogs' directory relative to
    the current file's directory. It constructs the module path for each file
    and yields it.

    Yields:
        The module path for each Python file in the 'cogs' directory

    Raises:
        FileNotFoundError: If the cogs directory doesn't exist
        OSError: If there's an error accessing the cogs directory
    """
    logger.debug(f"Starting extension discovery from HERE={HERE}")
    cogs_path = pathlib.Path(HERE) / "cogs"
    logger.debug(f"Looking for cogs in: {cogs_path}")

    if not cogs_path.exists():
        logger.error(f"Cogs directory not found: {cogs_path}")
        raise FileNotFoundError(f"Cogs directory not found: {cogs_path}")

    try:
        files = list(cogs_path.rglob("*.py"))
        logger.debug("Successfully initialized file search")

        for file in files:
            if file.name == "__init__.py":
                continue

            # Get path relative to the cogs directory
            relative_path = file.relative_to(pathlib.Path(HERE))
            extension_path = str(relative_path)[:-3].replace(os.sep, ".")
            logger.debug(f"Found extension file: {file}")
            logger.debug(f"Converting to module path: {extension_path}")
            yield extension_path

    except Exception as e:
        logger.error(f"Error discovering extensions: {e}")
        logger.exception("Extension discovery failed")
        raise


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
    logger.debug(f"Starting async extension discovery from HERE={HERE}")
    cogs_path = pathlib.Path(HERE) / "cogs"
    logger.debug(f"Looking for cogs in: {cogs_path}")

    if not cogs_path.exists():
        logger.error(f"Cogs directory not found: {cogs_path}")
        raise FileNotFoundError(f"Cogs directory not found: {cogs_path}")

    try:
        # Get list of all .py files in cogs directory
        files = list(cogs_path.rglob("*.py"))
        logger.debug(f"Found files: {files}")
        logger.debug("Successfully initialized async file search")

        for file in files:
            # Skip __init__.py files
            if file.name == "__init__.py":
                continue

            # Verify file exists and is readable
            try:
                async with aiofiles.open(file) as f:
                    # Just check if we can open it
                    await f.read(1)

                # Get path relative to the cogs directory
                relative_path = file.relative_to(pathlib.Path(HERE))
                extension_path = str(relative_path)[:-3].replace(os.sep, ".")

                logger.debug(f"Found extension file: {file}")
                logger.debug(f"Converting to module path: {extension_path}")
                yield extension_path
                await logger.complete()

            except OSError as e:
                logger.warning(f"Skipping inaccessible extension file {file}: {e}")
                continue

    except Exception as e:
        logger.error(f"Error discovering extensions: {e}")
        logger.exception("Extension discovery failed")
        raise

    logger.debug("Completed async extension discovery")
    await logger.complete()


async def load_extensions(bot: Any, extensions_list: list[str]) -> None:
    """Load a list of extensions into the bot.

    Args:
        bot: The Discord bot instance
        extensions_list: List of extension module paths to load

    Raises:
        Exception: If an extension fails to load
    """
    for ext in extensions_list:
        try:
            logger.debug(f"Loading extension: {ext}")
            await bot.load_extension(ext)
            logger.info(f"Successfully loaded extension: {ext}")
        except Exception as e:
            logger.error(f"Failed to load extension {ext}: {e}")
            logger.exception(f"Extension {ext} failed to load")
            raise


async def reload_extension(bot: Any, extension: str) -> None:
    """Reload a specific extension.

    Args:
        bot: The Discord bot instance
        extension: The extension module path to reload

    Raises:
        Exception: If the extension fails to reload
    """
    try:
        logger.debug(f"Reloading extension: {extension}")
        await bot.reload_extension(extension)
        logger.info(f"Successfully reloaded extension: {extension}")
    except Exception as e:
        logger.error(f"Failed to reload extension {extension}: {e}")
        logger.exception(f"Extension {extension} failed to reload")
        raise


async def unload_extension(bot: Any, extension: str) -> None:
    """Unload a specific extension.

    Args:
        bot: The Discord bot instance
        extension: The extension module path to unload

    Raises:
        Exception: If the extension fails to unload
    """
    try:
        logger.debug(f"Unloading extension: {extension}")
        await bot.unload_extension(extension)
        logger.info(f"Successfully unloaded extension: {extension}")
    except Exception as e:
        logger.error(f"Failed to unload extension {extension}: {e}")
        logger.exception(f"Extension {extension} failed to unload")
        raise
