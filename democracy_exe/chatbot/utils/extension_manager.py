# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Extension management utilities.

This module provides utilities for managing Discord bot extensions without circular dependencies.

<structlog_practices>
    <concurrency>
        - Use contextvars instead of direct logger binding for thread/async-safe context
        - Clear context at start and end of each method using clear_contextvars()
        - Bind context using bind_contextvars() for thread/async safety
        - Context is automatically isolated between different async tasks
        - Avoid using bind() directly as it's not thread/async-safe
    </concurrency>

    <context_management>
        - Clear context at start of each major operation
        - Use bound_contextvars context manager for temporary bindings
        - Ensure context is cleaned up after operations complete
        - Add relevant context for dependency resolution steps
        - Include error context for debugging dependency issues
    </context_management>

    <logging_patterns>
        - Use debug level for dependency resolution steps
        - Use info level for successful operations
        - Use error level for dependency conflicts
        - Include relevant extension and dependency context
        - Track visited and ordered states for debugging
    </logging_patterns>

    <example_usage>
        # Clear existing context before operation
        structlog.contextvars.clear_contextvars()

        # Bind context for operation
        structlog.contextvars.bind_contextvars(
            operation="load_order",
            extensions=extensions,
            dependencies=dependencies
        )

        try:
            # Perform operation
            result = process_extensions()

            # Log success with additional context
            logger.info("Operation succeeded", result=result)
        except Exception as e:
            # Log error with full context
            logger.error("Operation failed", error=str(e))
            raise
        finally:
            # Clean up context
            structlog.contextvars.clear_contextvars()
</structlog_practices>
"""
from __future__ import annotations

import asyncio
import os
import pathlib

from typing import TYPE_CHECKING, Dict, List, Set

import structlog

from discord.ext import commands

from democracy_exe.aio_settings import aiosettings


logger = structlog.get_logger(__name__)


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
    # Clear any existing context
    structlog.contextvars.clear_contextvars()

    try:
        # Bind initial context
        structlog.contextvars.bind_contextvars(
            operation="extension_load_order",
            extensions=extensions
        )

        logger.debug("Starting extension load order resolution")

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

            # Update context with dependency information
            structlog.contextvars.bind_contextvars(
                dependencies=dependencies,
                core_extension=core_ext
            )
            logger.debug("Initialized dependencies")

        visited: set[str] = set()
        temp_visited: set[str] = set()
        ordered: list[str] = []

        def visit(ext: str) -> None:
            """Visit an extension and its dependencies recursively.

            Args:
                ext: Extension module path to visit

            Raises:
                ValueError: If circular dependency detected
            """
            # Update context for current visit
            structlog.contextvars.bind_contextvars(
                current_extension=ext,
                temp_visited=list(temp_visited),
                visited=list(visited),
                ordered=ordered
            )

            logger.debug("Visiting extension")

            if ext in temp_visited:
                logger.error("Circular dependency detected")
                raise ValueError(f"Circular dependency detected involving {ext}")

            if ext in visited:
                logger.debug("Extension already visited")
                return

            temp_visited.add(ext)

            # Update context with dependencies
            structlog.contextvars.bind_contextvars(
                current_dependencies=list(dependencies[ext])
            )
            logger.debug("Processing dependencies")

            for dep in dependencies[ext]:
                structlog.contextvars.bind_contextvars(
                    current_dependency=dep
                )
                logger.debug("Processing dependency")
                visit(dep)

            temp_visited.remove(ext)
            visited.add(ext)
            ordered.append(ext)
            logger.debug("Completed extension visit")

        # Visit all extensions
        for ext in extensions:
            if ext not in visited:
                structlog.contextvars.bind_contextvars(
                    new_extension=ext
                )
                logger.debug("Starting new extension visit")
                visit(ext)

        logger.info("Completed extension load order resolution", final_order=ordered)
        return ordered

    finally:
        # Always clean up context
        structlog.contextvars.clear_contextvars()


async def load_extension_with_retry(bot: Bot, extension: str, max_attempts: int = 3) -> None:
    """Load an extension with retries.

    Args:
        bot: The Discord bot instance
        extension: Extension module path to load
        max_attempts: Maximum number of attempts to load the extension

    Raises:
        RuntimeError: If extension fails to load after max attempts
    """
    # Clear any existing context
    structlog.contextvars.clear_contextvars()

    try:
        # Bind initial context
        structlog.contextvars.bind_contextvars(
            operation="load_extension_retry",
            extension=extension,
            max_attempts=max_attempts,
            bot_id=bot.user.id if bot.user else None
        )

        attempt = 0
        while attempt < max_attempts:
            try:
                # Update attempt context
                structlog.contextvars.bind_contextvars(
                    attempt=attempt + 1,
                    remaining_attempts=max_attempts - attempt - 1,
                    backoff_seconds=attempt
                )

                await bot.load_extension(extension)
                logger.info("Extension loaded successfully")
                return

            except Exception as e:
                attempt += 1
                if attempt < max_attempts:
                    # Update error context
                    structlog.contextvars.bind_contextvars(
                        error=str(e),
                        error_type=type(e).__name__,
                        remaining_attempts=max_attempts - attempt
                    )
                    logger.warning("Extension load attempt failed, retrying")
                    await asyncio.sleep(1 * attempt)  # Exponential backoff
                else:
                    # Update final error context
                    structlog.contextvars.bind_contextvars(
                        error=str(e),
                        error_type=type(e).__name__,
                        total_attempts=attempt
                    )
                    logger.error("Extension load failed after all attempts")
                    raise RuntimeError(f"Failed to load extension {extension} after {max_attempts} attempts") from e

    finally:
        # Always clean up context
        structlog.contextvars.clear_contextvars()


async def load_extensions(bot: Bot, extension_list: list[str]) -> None:
    """Load a list of extensions into the bot.

    Args:
        bot: The Discord bot instance
        extension_list: List of extension module paths to load

    Raises:
        Exception: If loading an extension fails
    """
    # Clear any existing context
    structlog.contextvars.clear_contextvars()

    try:
        # Bind initial context
        structlog.contextvars.bind_contextvars(
            operation="load_extensions",
            extension_count=len(extension_list),
            bot_id=bot.user.id if bot.user else None
        )

        logger.debug("Starting extension loading")

        for idx, extension in enumerate(extension_list, 1):
            try:
                # Update context for current extension
                structlog.contextvars.bind_contextvars(
                    current_extension=extension,
                    progress=f"{idx}/{len(extension_list)}"
                )

                await bot.load_extension(extension)
                logger.info("Extension loaded successfully")

            except Exception as e:
                # Update error context
                structlog.contextvars.bind_contextvars(
                    error=str(e),
                    error_type=type(e).__name__,
                    failed_extension=extension
                )
                logger.error("Extension load failed")
                raise

    finally:
        # Always clean up context
        structlog.contextvars.clear_contextvars()


async def reload_extension(bot: Bot, extension: str) -> None:
    """Reload a specific extension.

    Args:
        bot: The Discord bot instance
        extension: Extension module path to reload

    Raises:
        Exception: If reloading the extension fails
    """
    # Clear any existing context
    structlog.contextvars.clear_contextvars()

    try:
        # Bind initial context
        structlog.contextvars.bind_contextvars(
            operation="reload_extension",
            extension=extension,
            bot_id=bot.user.id if bot.user else None
        )

        logger.debug("Starting extension reload")

        try:
            await bot.reload_extension(extension)
            logger.info("Extension reloaded successfully")

        except Exception as e:
            # Update error context
            structlog.contextvars.bind_contextvars(
                error=str(e),
                error_type=type(e).__name__
            )
            logger.error("Extension reload failed")
            raise

    finally:
        # Always clean up context
        structlog.contextvars.clear_contextvars()


async def unload_extension(bot: Bot, extension: str) -> None:
    """Unload a specific extension.

    Args:
        bot: The Discord bot instance
        extension: Extension module path to unload

    Raises:
        Exception: If unloading the extension fails
    """
    # Clear any existing context
    structlog.contextvars.clear_contextvars()

    try:
        # Bind initial context
        structlog.contextvars.bind_contextvars(
            operation="unload_extension",
            extension=extension,
            bot_id=bot.user.id if bot.user else None
        )

        logger.debug("Starting extension unload")

        try:
            await bot.unload_extension(extension)
            logger.info("Extension unloaded successfully")

        except Exception as e:
            # Update error context
            structlog.contextvars.bind_contextvars(
                error=str(e),
                error_type=type(e).__name__
            )
            logger.error("Extension unload failed")
            raise

    finally:
        # Always clean up context
        structlog.contextvars.clear_contextvars()
