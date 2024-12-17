# pyright: reportAttributeAccessIssue=false
"""LangChain integration utilities for Discord bot.

This module handles AI/LangChain integration for the Discord bot.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import discord

from discord import DMChannel, Message, Thread
from langchain_core.messages import HumanMessage
from loguru import logger

from democracy_exe.agentic.workflows.react.graph import graph as memgraph
from democracy_exe.ai.graphs import AgentState
from democracy_exe.chatbot.utils.message_utils import format_inbound_message, get_session_id, prepare_agent_input


async def get_thread(message: Message) -> Thread | DMChannel:
    """Get or create a thread for the message.

    Args:
        message: The Discord message

    Returns:
        Thread or DMChannel for responses
    """
    if isinstance(message.channel, DMChannel):
        return message.channel

    thread = await message.channel.create_thread(
        name="Response",
        message=message
    )
    return thread


async def stream_bot_response(
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
        ValueError: If response generation fails
    """
    try:
        if user_input is None:
            raise ValueError("user_input must be provided")

        response = graph.invoke(user_input)
        if isinstance(response, dict) and "messages" in response:
            messages = response.get("messages", [])
            return "".join(msg.content for msg in messages if hasattr(msg, 'content'))
        raise ValueError("No response generated")
    except Exception as e:
        logger.error(f"Error streaming bot response: {e}")
        raise
