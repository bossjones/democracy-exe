# pylint: disable=too-many-function-args
# mypy: disable-error-code="arg-type, var-annotated, list-item, no-redef, truthy-bool, return-value"
# pyright: reportPrivateImportUsage=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
from __future__ import annotations

import logging

from typing import List, Optional

import discord

# from loguru import logger
import structlog

from discord import Message as DiscordMessage


logger = structlog.get_logger(__name__)

from democracy_exe.base import DemocracyMessage
from democracy_exe.constants import INACTIVATE_THREAD_PREFIX, MAX_CHARS_PER_REPLY_MSG


def discord_message_to_message(message: discord.Message) -> DemocracyMessage | None:
    """Convert a Discord message to a DemocracyMessage.

    Args:
        message: The Discord message to convert.

    Returns:
        DemocracyMessage if conversion is successful, None otherwise.
    """
    if (
        message.type == discord.MessageType.thread_starter_message
        and message.reference.cached_message
        and len(message.reference.cached_message.embeds) > 0
        and len(message.reference.cached_message.embeds[0].fields) > 0
    ):
        field = message.reference.cached_message.embeds[0].fields[0]
        if field.value:
            return DemocracyMessage(user=field.name, text=field.value)
    elif message.content:
        return DemocracyMessage(user=message.author.name, text=message.content)
    return None


def split_into_shorter_messages(message: str) -> list[str]:
    """Split a message into shorter messages that fit within Discord's character limit.

    Args:
        message: The message to split.

    Returns:
        List of message chunks that fit within the character limit.
    """
    return [message[i : i + MAX_CHARS_PER_REPLY_MSG] for i in range(0, len(message), MAX_CHARS_PER_REPLY_MSG)]


def is_last_message_stale(interaction_message: DiscordMessage, last_message: DiscordMessage, bot_id: str) -> bool:
    """Check if the last message in a thread is stale.

    Args:
        interaction_message: The message that triggered the interaction.
        last_message: The last message in the thread.
        bot_id: The ID of the bot.

    Returns:
        True if the last message is stale, False otherwise.
    """
    return (
        last_message
        and last_message.id != interaction_message.id
        and last_message.author
        and last_message.author.id != bot_id
    )


async def close_thread(thread: discord.Thread) -> None:
    """Close a Discord thread.

    Args:
        thread: The thread to close.
    """
    await thread.edit(name=INACTIVATE_THREAD_PREFIX)
    await thread.send(
        embed=discord.Embed(
            description="**Thread closed** - Context limit reached, closing...",
            color=discord.Color.blue(),
        )
    )
    await thread.edit(archived=True, locked=True)
