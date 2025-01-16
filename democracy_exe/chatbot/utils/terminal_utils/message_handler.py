# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Terminal message handling utilities for the chatbot.

This module provides utilities for handling and processing messages in the terminal bot.
It includes functionality for:
- Message formatting and conversion
- Rich text output handling
- Message type validation and conversion
- Stream processing of messages

Key Components:
    - MessageHandler: Core class for message processing
    - MessageFormatter: Handles message formatting and display
    - StreamProcessor: Manages message streaming

Dependencies:
    - langchain_core.messages: For message type definitions
    - rich: For terminal formatting
    - structlog: For logging

Example:
    ```python
    handler = MessageHandler()
    formatted_msg = await handler.format_message(user_input)
    async for chunk in handler.stream_response(response):
        print(chunk)
    ```

Note:
    This module is designed to work with both sync and async code paths,
    prioritizing async operations for better performance.
"""
from __future__ import annotations

import asyncio

from collections.abc import AsyncGenerator
from typing import Any, Optional, Union

import structlog

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rich import print as rprint


logger = structlog.get_logger(__name__)


class MessageHandler:
    """Handles message processing and formatting for the terminal bot."""

    def __init__(self) -> None:
        """Initialize the message handler."""
        self._logger = logger.bind(module="MessageHandler")

    async def format_message(self, content: str) -> HumanMessage:
        """Format user input as a HumanMessage.

        Args:
            content: Raw user input string

        Returns:
            HumanMessage: Formatted message object
        """
        return HumanMessage(content=content)

    async def format_response(self, message: BaseMessage) -> None:
        """Format and display an AI response message.

        Args:
            message: The message to format and display
        """
        try:
            if isinstance(message, AIMessage):
                rprint(f"[bold blue]AI:[/bold blue] {message.content}")
                self._logger.info("AI response", content=message.content)
            else:
                message.pretty_print()
        except Exception as e:
            self._logger.error("Error formatting response", error=str(e))
            raise

    async def create_input_dict(self, message: HumanMessage) -> dict[str, list[BaseMessage]]:
        """Create input dictionary for graph processing.

        Args:
            message: The human message to process

        Returns:
            dict: Input dictionary with messages list
        """
        return {"messages": [message]}

    async def stream_chunks(
        self,
        chunks: AsyncGenerator[Any, None]
    ) -> AsyncGenerator[str, None]:
        """Stream and format message chunks.

        Args:
            chunks: Generator of message chunks

        Yields:
            Formatted message chunks
        """
        try:
            async for chunk in chunks:
                if isinstance(chunk, dict) and "messages" in chunk:
                    message = chunk["messages"][-1]
                    await self.format_response(message)
                    yield str(message.content)
        except Exception as e:
            self._logger.error("Error streaming chunks", error=str(e))
            raise
