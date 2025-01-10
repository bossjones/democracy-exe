# pyright: reportAttributeAccessIssue=false
"""Message handling utilities for Discord bot.

This module provides utilities for handling Discord messages, including
formatting, thread management, and session handling.
"""
from __future__ import annotations

from typing import Any, Optional, Union, cast

import discord
import structlog

from discord import DMChannel, Guild, Member, Message, TextChannel, Thread, User
from discord.abc import Messageable
from langchain_core.messages import AIMessage, HumanMessage


logger = structlog.get_logger(__name__)


def format_inbound_message(message: Message) -> HumanMessage:
    """Format a Discord message into a HumanMessage for LangGraph processing.

    Args:
        message: The Discord message to format

    Returns:
        A formatted message ready for LangGraph processing

    Raises:
        ValueError: If message formatting fails
    """
    try:
        guild = cast(Optional[Guild], message.guild)
        channel = cast(Union[TextChannel, DMChannel], message.channel)
        author = cast(Union[Member, User], message.author)

        # Build metadata string
        guild_str = f"guild={guild.name}" if guild else ""
        content = f"""<discord {guild_str} channel={channel} author={author!r}>
        {message.content}
        </discord>"""

        logger.debug(f"Formatted message content: {content}")

        if author.global_name is None:
            raise ValueError("Failed to format message: author has no global name")

        return HumanMessage(
            content=content,
            name=str(author.global_name),
            id=str(message.id)
        )
    except Exception as e:
        logger.error(f"Error formatting message: {e}")
        raise ValueError(f"Failed to format message: {e}")


async def get_or_create_thread(message: Message) -> Thread | DMChannel | None:
    """Get or create a thread for the message.

    Args:
        message: The message to get/create a thread for

    Returns:
        Either a Thread object for server channels or DMChannel for direct messages

    Raises:
        discord.HTTPException: If thread creation fails
    """
    try:
        channel = message.channel

        # If this is a DM channel, just return it directly
        if isinstance(channel, DMChannel):
            return channel

        # For regular channels, create a thread
        if isinstance(channel, (TextChannel, Thread)):
            try:
                return await channel.create_thread(name="Response", message=message)
            except discord.HTTPException as e:
                logger.error(f"Failed to create thread: {e}")
                if not hasattr(e, 'status'):
                    e.status = 400
                raise

        return None
    except Exception as e:
        logger.error(f"Error getting/creating thread: {e}")
        raise


def get_session_id(message: Message | Thread) -> str:
    """Generate a session ID for the given message.

    Args:
        message: The message or event dictionary

    Returns:
        The generated session ID

    Notes:
        - If the message is a direct message (DM), the session ID is based on the user ID
        - If the message is from a guild channel, the session ID is based on the channel ID
    """
    try:
        if isinstance(message, Thread):
            starter_message = cast(Message, message.starter_message)
            if starter_message is None:
                raise ValueError("Thread has no starter message")
            channel = cast(Union[DMChannel, Any], starter_message.channel)
            is_dm = str(channel.type) == "private"
            user_id = starter_message.author.id
            channel_id = channel.name if isinstance(channel, Thread) else channel.id
        else:
            channel = cast(Union[DMChannel, Any], message.channel)
            is_dm = str(channel.type) == "private"
            user_id = message.author.id
            channel_id = channel.id

        if user_id is None or channel_id is None:
            raise ValueError("Could not determine user_id or channel_id")

        return f"discord_{user_id}" if is_dm else f"discord_{channel_id}"
    except Exception as e:
        logger.error(f"Error generating session ID: {e}")
        return f"discord_fallback_{discord.utils.utcnow().timestamp()}"


def prepare_agent_input(
    message: Message | Thread,
    user_real_name: str,
    surface_info: dict[str, Any]
) -> dict[str, Any]:
    """Prepare the agent input from the incoming Discord message.

    Args:
        message: The Discord message containing the user input
        user_real_name: The real name of the user who sent the message
        surface_info: The surface information related to the message

    Returns:
        The input dictionary to be sent to the agent

    Raises:
        ValueError: If message processing fails
    """
    try:
        if isinstance(message, Thread):
            starter_message = cast(Message, message.starter_message)
            if starter_message is None:
                raise ValueError("Thread has no starter message")
            content = starter_message.content
            attachments = starter_message.attachments
        else:
            content = message.content
            attachments = message.attachments

        if content is None:
            raise ValueError("Message has no content")

        agent_input = {
            "user name": user_real_name,
            "message": content,
            "surface_info": surface_info
        }

        if attachments:
            for attachment in attachments:
                logger.debug(f"Processing attachment: {attachment}")
                agent_input["file_name"] = attachment.filename
                if attachment.content_type and attachment.content_type.startswith("image/"):
                    agent_input["image_url"] = attachment.url

        return agent_input
    except Exception as e:
        logger.error(f"Error preparing agent input: {e}")
        raise ValueError(f"Failed to prepare agent input: {e}")
