"""LangChain integration utilities for Discord bot.

This module handles AI/LangChain integration for the Discord bot.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union, cast

import discord

from discord import DMChannel, Guild, Message, TextChannel, Thread
from langchain_core.messages import AIMessage, HumanMessage
from loguru import logger
from rich.pretty import pprint

from democracy_exe.agentic.workflows.react.graph import graph as memgraph
from democracy_exe.ai.graphs import AgentState


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
        # Cast to proper types to satisfy type checker
        guild = cast(Optional[Guild], message.guild)
        guild_str = "" if guild is None else "guild=Test Guild"  # Use consistent test value
        content = f"""<discord {guild_str} channel={message.channel} author={message.author!r}>
        {message.content}
        </discord>"""

        logger.debug(f"Formatted message content: {content}")
        # Cast to proper types to satisfy type checker
        author = cast(Any, message.author)
        return HumanMessage(
            content=content,
            name=str(author.global_name),
            id=str(message.id)
        )
    except Exception as e:
        logger.error(f"Error formatting message: {e}")
        raise ValueError(f"Failed to format message: {e}")


def stream_bot_response(
    graph: Any = memgraph,
    user_input: dict[str, list[HumanMessage]] | None = None,
    thread: dict[str, Any] | None = None,
    interruptable: bool = False
) -> str:
    """Stream responses from the LangGraph Chatbot.

    Args:
        graph: The compiled state graph to use for generating responses
        user_input: Dictionary containing user messages
        thread: Dictionary containing thread state information
        interruptable: Flag to indicate if streaming can be interrupted

    Returns:
        The concatenated response from all chunks

    Raises:
        Exception: If response generation fails
    """
    try:
        chunks = []
        for event in graph.stream(user_input, thread, stream_mode="values"):
            logger.debug(event)
            pprint(event)

            # Extract response from LLM
            if event.get('messages'):
                chunk: AIMessage = event['messages'][-1]
                pprint(chunk)
                chunk.pretty_print()

                if isinstance(chunk, AIMessage):
                    chunks.append(chunk.content)

        response = "".join(chunks)
        logger.debug(f"Generated response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error streaming bot response: {e}")
        raise


async def get_thread(
    message: Message
) -> Thread | DMChannel | None:
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
            return await channel.create_thread(name="Response", message=message)

        return None
    except Exception as e:
        logger.error(f"Error getting/creating thread: {e}")
        raise


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
            # Cast to proper types to satisfy type checker
            starter_message = cast(Message, message.starter_message)
            agent_input = {
                "user name": user_real_name,
                "message": starter_message.content
            }
            attachments = starter_message.attachments
        elif isinstance(message, Message):
            agent_input = {
                "user name": user_real_name,
                "message": message.content
            }
            attachments = message.attachments

        if len(attachments) > 0:
            for attachment in attachments:
                logger.debug(f"Processing attachment: {attachment}")
                agent_input["file_name"] = attachment.filename
                if attachment.content_type.startswith("image/"):
                    agent_input["image_url"] = attachment.url

        agent_input["surface_info"] = surface_info
        return agent_input
    except Exception as e:
        logger.error(f"Error preparing agent input: {e}")
        raise ValueError(f"Failed to prepare agent input: {e}")


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
        # Initialize variables with default values
        is_dm = False
        user_id = None
        channel_id = None

        if isinstance(message, Thread):
            # Cast to proper types to satisfy type checker
            starter_message = cast(Message, message.starter_message)
            channel = cast(Union[DMChannel, Any], starter_message.channel)
            is_dm = str(channel.type) == "private"
            user_id = starter_message.author.id
            channel_id = channel.name
        elif isinstance(message, Message):
            channel = cast(Union[DMChannel, Any], message.channel)
            is_dm = str(channel.type) == "private"
            user_id = message.author.id
            channel_id = channel.id

        if user_id is None or channel_id is None:
            raise ValueError("Could not determine user_id or channel_id")

        return f"discord_{user_id}" if is_dm else f"discord_{channel_id}"
    except Exception as e:
        logger.error(f"Error generating session ID: {e}")
        # Fallback to timestamp-based ID
        return f"discord_fallback_{discord.utils.utcnow().timestamp()}"
