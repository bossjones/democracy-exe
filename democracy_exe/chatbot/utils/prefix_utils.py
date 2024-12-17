"""Prefix handling utilities for Discord bot.

This module handles command prefix management for the Discord bot.
"""
from __future__ import annotations

from typing import Any, List, Optional, Union, cast

import discord

from discord import DMChannel, Guild, Message
from discord.ext import commands
from loguru import logger

from democracy_exe.aio_settings import aiosettings


def get_guild_prefix(bot: Any, guild_id: int) -> str:
    """Get the prefix for a specific guild.

    Args:
        bot: The bot instance
        guild_id: The guild ID

    Returns:
        The guild's prefix or default prefix
    """
    try:
        if not hasattr(bot, 'prefixes'):
            raise AttributeError("Bot has no prefixes attribute")
        return bot.prefixes.get(guild_id, [aiosettings.prefix])[0]
    except Exception as e:
        logger.error(f"Error getting guild prefix: {e}")
        return aiosettings.prefix


async def get_prefix(bot: Any, message: Message) -> Any:
    """Retrieve the command prefix for the bot based on the message context.

    Args:
        bot: The instance of the bot
        message: The message object from Discord

    Returns:
        The command prefix to be used for the bot
    """
    logger.info(f"Getting prefix for message: {message.id}")
    try:
        # Cast to proper types to satisfy type checker
        channel = cast(Union[DMChannel, Any], message.channel)
        prefix = (
            [aiosettings.prefix]
            if isinstance(channel, DMChannel)
            else [get_guild_prefix(bot, cast(Guild, message.guild).id)]
        )
        logger.debug(f"Using prefix: {prefix}")
        await logger.complete()
        base = [f"<@!{bot.user.id}> ", f"<@{bot.user.id}> "]
        prefixes = [aiosettings.prefix] if isinstance(channel, DMChannel) else bot.prefixes.get(cast(Guild, message.guild).id, [aiosettings.prefix])
        base.extend(prefixes)
        return base
    except Exception as e:
        logger.error(f"Error getting prefix: {e}")
        # Fallback to default prefix
        return commands.when_mentioned_or(aiosettings.prefix)(bot, message)


def _prefix_callable(bot: Any, msg: Message) -> list[str]:
    """Generate a list of command prefixes for the bot.

    This function generates a list of command prefixes for the bot based on the
    message context. If the message is from a direct message (DM) channel, it
    includes the bot's user ID mentions and default prefixes. If the message is
    from a guild (server) channel, it includes the bot's user ID mentions and
    the guild-specific prefixes.

    Args:
        bot: The instance of the bot
        msg: The message object from Discord

    Returns:
        List of command prefixes to be used for the bot
    """
    try:
        user_id = bot.user.id
        base = [f"<@!{user_id}> ", f"<@{user_id}> "]

        # Cast to proper types to satisfy type checker
        guild = cast(Optional[Guild], msg.guild)
        if guild is None:
            base.extend(("!", "?"))
        else:
            base.extend(bot.prefixes.get(guild.id, ["?", "!"]))

        return base
    except Exception as e:
        logger.error(f"Error in prefix_callable: {e}")
        # Fallback to default prefixes
        return ["!", "?"]


async def update_guild_prefix(
    bot: Any, guild_id: int, new_prefix: str
) -> None:
    """Update the command prefix for a specific guild.

    Args:
        bot: The instance of the bot
        guild_id: The ID of the guild to update
        new_prefix: The new prefix to set

    Raises:
        ValueError: If the new prefix is invalid
    """
    try:
        if not new_prefix or len(new_prefix) > 10:
            raise ValueError("Invalid prefix length")

        if guild_id in bot.prefixes:
            bot.prefixes[guild_id] = [new_prefix]
            logger.info(f"Updated prefix for guild {guild_id} to {new_prefix}")
        else:
            logger.warning(f"Guild {guild_id} not found in prefix cache")

    except Exception as e:
        logger.error(f"Error updating guild prefix: {e}")
        raise


def get_prefix_display(
    bot: Any, guild: Guild | None = None
) -> str:
    """Get a display string for the current prefix(es).

    Args:
        bot: The instance of the bot
        guild: The guild to get prefixes for (None for DM prefixes)

    Returns:
        A formatted string showing the current prefix(es)
    """
    try:
        if guild is None:
            prefixes = ["!", "?"]
        else:
            prefixes = bot.prefixes.get(guild.id, ["?", "!"])

        if len(prefixes) == 1:
            return f"Current prefix is: {prefixes[0]}"
        else:
            return f"Current prefixes are: {', '.join(prefixes)}"
    except Exception as e:
        logger.error(f"Error getting prefix display: {e}")
        return "Default prefixes are: ! ?"
