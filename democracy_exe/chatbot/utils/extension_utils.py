# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Utility functions for managing Discord bot extensions."""
from __future__ import annotations

import asyncio
import importlib
import logging
import os
import pathlib
import sys
import traceback

from typing import Any, Dict, List, Optional, Tuple

import discord
import structlog

from discord.ext import commands


HERE = os.path.dirname(os.path.dirname(__file__))

logger = structlog.get_logger(__name__)

async def load_extension_with_retry(
    bot: commands.Bot,
    extension: str,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> None:
    """Load a Discord bot extension with retries.

    Args:
        bot: The Discord bot instance
        extension: Name of the extension to load
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Raises:
        commands.ExtensionError: If extension fails to load after retries
    """
    for attempt in range(max_retries):
        try:
            await bot.load_extension(extension)
            logger.info(f"Successfully loaded extension {extension}")
            return
        except commands.ExtensionError as e:
            if attempt == max_retries - 1:
                logger.error(f"Failed to load extension {extension} after {max_retries} attempts")
                raise
            logger.warning(f"Failed to load extension {extension}, attempt {attempt + 1}/{max_retries}")
            await asyncio.sleep(retry_delay)

def get_extension_load_order() -> list[str]:
    """Get the ordered list of extensions to load.

    Returns:
        List of extension names in load order
    """
    return [
        "democracy_exe.chatbot.cogs.twitter",
        # Add other extensions here in desired load order
    ]

async def aio_extensions(bot: commands.Bot) -> None:
    """Load all extensions asynchronously.

    Args:
        bot: The Discord bot instance
    """
    extensions = get_extension_load_order()
    for extension in extensions:
        try:
            await load_extension_with_retry(bot, extension)
        except commands.ExtensionError as e:
            logger.error(f"Failed to load extension {extension}: {e}")
