# pyright: reportAttributeAccessIssue=false
"""LangChain integration utilities for Discord bot.

This module handles AI/LangChain integration for the Discord bot.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import discord

from discord import Message, Thread
from langchain_core.messages import HumanMessage
from loguru import logger

from democracy_exe.agentic.workflows.react.graph import graph as memgraph
from democracy_exe.ai.graphs import AgentState
from democracy_exe.chatbot.utils.message_utils import format_inbound_message, get_session_id, prepare_agent_input


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
        ValueError: If response generation fails
    """
    try:
        if user_input is None or thread is None:
            raise ValueError("user_input and thread must be provided")

        response = graph.invoke(user_input, thread)
        if not response or "response" not in response:
            raise ValueError("No response generated")

        return response["response"]
    except Exception as e:
        logger.error(f"Error streaming bot response: {e}")
        raise
