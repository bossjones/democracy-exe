# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Terminal stream handling utilities for the chatbot.

This module provides utilities for managing streaming responses in the terminal bot.
It includes functionality for:
- Stream management and processing
- Chunk handling and formatting
- Interruptible streams
- Stream state management

Key Components:
    - StreamHandler: Core class for stream management
    - StreamConfig: Configuration for stream behavior
    - StreamProcessor: Processes and formats stream chunks

Dependencies:
    - langchain_core.messages: For message types
    - langgraph.graph: For graph streaming
    - structlog: For logging

Example:
    ```python
    handler = StreamHandler()
    async with handler.stream_context():
        async for chunk in handler.process_stream(graph, input_dict):
            print(chunk)
    ```

Note:
    This module is designed to work with LangGraph's streaming capabilities
    and provides both interruptible and continuous streaming modes.
"""
from __future__ import annotations

import asyncio

from collections.abc import AsyncGenerator
from typing import Any, Optional, Union

import structlog

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from democracy_exe.chatbot.utils.terminal_utils.message_handler import MessageHandler


logger = structlog.get_logger(__name__)


class StreamHandler:
    """Handles stream processing for the terminal bot."""

    def __init__(self) -> None:
        """Initialize the stream handler."""
        self._logger = logger.bind(module="StreamHandler")
        self._message_handler = MessageHandler()
        self._interrupt_event = asyncio.Event()

    async def process_stream(
        self,
        graph: CompiledStateGraph,
        user_input: dict[str, list[BaseMessage]],
        config: RunnableConfig | None = None,
        interruptable: bool = False
    ) -> AsyncGenerator[str, None]:
        """Process a stream from the graph.

        Args:
            graph: The compiled state graph
            user_input: Dictionary containing user messages
            config: Optional runnable configuration
            interruptable: Whether the stream can be interrupted

        Yields:
            Processed message chunks
        """
        try:
            # Initial stream processing
            for event in graph.stream(user_input, config, stream_mode="values"):
                if interruptable and self._interrupt_event.is_set():
                    break

                async for chunk in self._message_handler.stream_chunks([event]):
                    yield chunk

            if interruptable and self._interrupt_event.is_set():
                # Log interruption
                self._logger.info("Stream interrupted")
                # Get user feedback
                user_approval = await asyncio.to_thread(
                    input,
                    "Do you want to call the tool? (yes[y]/no[n]): "
                )

                if user_approval.lower() in ("yes", "y"):
                    # Continue stream if approved
                    for event in graph.stream(None, config, stream_mode="values"):
                        async for chunk in self._message_handler.stream_chunks([event]):
                            yield chunk
                else:
                    yield "Operation cancelled by user."

        except Exception as e:
            self._logger.error("Error processing stream", error=str(e))
            raise

    def interrupt(self) -> None:
        """Interrupt the current stream."""
        self._interrupt_event.set()

    def reset(self) -> None:
        """Reset the interrupt state."""
        self._interrupt_event.clear()

    async def __aenter__(self) -> StreamHandler:
        """Enter async context.

        Returns:
            StreamHandler: This instance
        """
        self.reset()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self.reset()
