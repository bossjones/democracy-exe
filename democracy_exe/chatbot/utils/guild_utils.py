"""Guild utility functions for Discord bot.

This module handles guild-related functionality like data preloading and configuration.
"""
from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from democracy_exe.factories import guild_factory


async def preload_guild_data() -> dict[int, dict[str, Any]]:
    """Preload guild data.

    This function initializes and returns a dictionary containing guild data.
    Each guild is represented by its ID and contains a dictionary with the guild's prefix.

    Returns:
        A dictionary where the keys are guild IDs and the values are dictionaries
        containing guild-specific data, such as the prefix.
    """
    logger.info("Preloading guild data...")
    guilds = [guild_factory.Guild()]
    await logger.complete()
    return {guild.id: {"prefix": guild.prefix} for guild in guilds}
