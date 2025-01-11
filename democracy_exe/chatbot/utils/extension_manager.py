"""Extension management utilities.

This module provides utilities for managing Discord bot extensions without circular dependencies.
"""
from __future__ import annotations

import asyncio
import os
import pathlib

from typing import TYPE_CHECKING, Dict, List, Set

import structlog

from discord.ext import commands


logger = structlog.get_logger(__name__)

from democracy_exe.aio_settings import aiosettings


if TYPE_CHECKING:
    from discord.ext.commands import Bot


def get_extension_load_order(extensions: list[str], dependencies: dict[str, set[str]] | None = None) -> list[str]:
    """Get the order in which extensions should be loaded based on dependencies.

    Args:
        extensions: List of extension module paths
        dependencies: Optional dictionary mapping extensions to their dependencies.
                     If not provided, core will be made a dependency of all other extensions.

    Returns:
        List of extension module paths in dependency order

    Raises:
        ValueError: If circular dependencies are detected
    """
    # Track dependencies and visited nodes for cycle detection
    if dependencies is None:
        dependencies = {ext: set() for ext in extensions}
        # Core should be loaded first
        core_ext = next((ext for ext in extensions if "core" in ext), None)
        if core_ext:
            # Make other extensions depend on core
            for ext in extensions:
                if ext != core_ext:
                    dependencies[ext].add(core_ext)

    visited: set[str] = set()
    temp_visited: set[str] = set()
    ordered: list[str] = []

    def visit(ext: str) -> None:
        """Visit an extension and its dependencies recursively.

        Args:
            ext: Extension module path to visit

        Raises:
            ValueError: If circular dependencies are detected
        """
        if ext in temp_visited:
            raise ValueError(f"Circular dependency detected involving {ext}")
        if ext in visited:
            return

        temp_visited.add(ext)

        for dep in dependencies[ext]:
            visit(dep)

        temp_visited.remove(ext)
        visited.add(ext)
        ordered.append(ext)

    # Visit all extensions
    for ext in extensions:
        if ext not in visited:
            visit(ext)

    return ordered


async def load_extension_with_retry(bot: Bot, extension: str, max_attempts: int = 3) -> None:
    """Load an extension with retries.

    Args:
        bot: The Discord bot instance
        extension: Extension module path to load
        max_attempts: Maximum number of attempts to load the extension

    Raises:
        RuntimeError: If extension fails to load after max attempts
    """
    attempt = 0
    while attempt < max_attempts:
        try:
            await bot.load_extension(extension)
            logger.info(f"Loaded extension: {extension}")
            return
        except Exception as e:
            attempt += 1
            if attempt < max_attempts:
                logger.warning(f"Failed to load extension {extension} (attempt {attempt}/{max_attempts}): {e}")
                await asyncio.sleep(1 * attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to load extension {extension} after {max_attempts} attempts: {e}")
                raise RuntimeError(f"Failed to load extension {extension} after {max_attempts} attempts") from e


async def load_extensions(bot: Bot, extension_list: list[str]) -> None:
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
        except Exception as e:
            logger.error(f"Failed to load extension {extension}: {e}")
            raise


async def reload_extension(bot: Bot, extension: str) -> None:
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
    except Exception as e:
        logger.error(f"Failed to reload extension {extension}: {e}")
        raise


async def unload_extension(bot: Bot, extension: str) -> None:
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
    except Exception as e:
        logger.error(f"Failed to unload extension {extension}: {e}")
        raise
