"""Terminal bot implementation for the chatbot.

This module provides the main terminal bot implementation, integrating:
- Message handling and formatting
- Stream processing
- UI management
- Resource management
- LangGraph integration

The bot supports both streaming and non-streaming modes, with proper
resource management and graceful shutdown handling.

Key Components:
    - ThreadSafeTerminalBot: Main bot class
    - stream_terminal_bot: Streaming interface
    - invoke_terminal_bot: Non-streaming interface
    - go_terminal_bot: Main entry point

Example:
    ```python
    async def main():
        bot = ThreadSafeTerminalBot()
        async with bot:
            await bot.start()

    if __name__ == "__main__":
        asyncio.run(main())
    ```

Note:
    This implementation prioritizes thread safety and resource management
    while maintaining a clean interface for both streaming and non-streaming
    operations.
"""
from __future__ import annotations

import asyncio
import signal
import sys
import threading

from collections.abc import AsyncGenerator
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph

from democracy_exe.agentic.workflows.react.graph import graph as memgraph
from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.utils.resource_manager import ResourceLimits, ResourceManager
from democracy_exe.chatbot.utils.terminal_utils.message_handler import MessageHandler
from democracy_exe.chatbot.utils.terminal_utils.stream_handler import StreamHandler
from democracy_exe.chatbot.utils.terminal_utils.ui_manager import UIManager


logger = structlog.get_logger(__name__)


class BotState(Enum):
    """Enumeration of possible bot states."""
    RUNNING = auto()
    CLOSED = auto()


class ThreadSafeTerminalBot:
    """Thread-safe terminal bot implementation."""

    def __init__(self) -> None:
        """Initialize the terminal bot."""
        # Get resource limits from settings or use defaults
        limits = ResourceLimits(
            max_memory_mb=getattr(aiosettings, "max_memory_mb", 4096),
            max_tasks=getattr(aiosettings, "max_tasks", 100),
            max_response_size_mb=getattr(aiosettings, "max_response_size_mb", 1),
            max_buffer_size_kb=getattr(aiosettings, "max_buffer_size_kb", 64),
            task_timeout_seconds=getattr(aiosettings, "task_timeout_seconds", 30)
        )
        self._resource_manager = ResourceManager(limits=limits)
        self._shutdown_event = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self._state = BotState.CLOSED
        self._message_handler = MessageHandler()
        self._stream_handler = StreamHandler()
        self._ui_manager = UIManager()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._handle_signal)

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        logger.info("Received shutdown signal", signal=signum)
        if not self._shutdown_event.is_set():
            self._shutdown_event.set()

    async def _cleanup(self) -> None:
        """Clean up resources before shutdown."""
        try:
            await self._resource_manager.force_cleanup()
            for task in self._tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=1.0)
                    except (TimeoutError, asyncio.CancelledError):
                        pass
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))

    async def stream_terminal_bot(
        self,
        prompt: str,
        graph: CompiledStateGraph = memgraph,
        config: RunnableConfig | None = None,
        interruptable: bool = False
    ) -> AsyncGenerator[str, None]:
        """Stream bot responses in the terminal.

        Args:
            prompt: User input prompt
            graph: The graph to use for processing
            config: Optional runnable configuration
            interruptable: Whether the stream can be interrupted

        Yields:
            Bot response chunks

        Raises:
            RuntimeError: If memory limit is exceeded or task limit is reached
            ValueError: If no response is generated
        """
        # Check memory usage
        if not await self._resource_manager.check_memory():
            raise RuntimeError("Memory limit exceeded")

        # Create and track task
        task = asyncio.current_task()
        if task:
            await self._resource_manager.track_task(task)
            self._tasks.add(task)

        try:
            # Format message and create input dict
            message = await self._message_handler.format_message(prompt)
            input_dict = await self._message_handler.create_input_dict(message)

            # Process stream
            async with self._stream_handler as handler:
                async for chunk in handler.process_stream(
                    graph,
                    input_dict,
                    config,
                    interruptable
                ):
                    yield chunk

        except Exception as e:
            logger.error("Error streaming response", error=str(e))
            raise
        finally:
            if task and task in self._tasks:
                self._tasks.remove(task)

    async def invoke_terminal_bot(
        self,
        prompt: str,
        graph: CompiledStateGraph = memgraph,
        config: RunnableConfig | None = None
    ) -> tuple[str, list[dict[str, Any]]]:
        """Invoke the terminal bot with a prompt.

        Args:
            prompt: User input prompt
            graph: The graph to use for processing
            config: Optional runnable configuration

        Returns:
            Tuple of final answer and intermediate steps

        Raises:
            RuntimeError: If memory limit is exceeded or task limit is reached
            ValueError: If no response is generated
        """
        response_chunks = []
        async for chunk in self.stream_terminal_bot(prompt, graph, config):
            response_chunks.append(chunk)

        response = ''.join(response_chunks)
        if not response:
            raise ValueError("No response generated")

        return response, []  # Empty list for intermediate steps

    async def start(self) -> None:
        """Start the terminal bot."""
        try:
            logger.info("Starting terminal bot")
            self._state = BotState.RUNNING

            async with self._ui_manager as ui:
                await ui.display_welcome()

                while True:
                    user_input = await ui.get_input()

                    if user_input.lower() == 'quit':
                        await ui.display_goodbye()
                        break

                    try:
                        async for chunk in self.stream_terminal_bot(user_input):
                            await ui.display_response(chunk)
                    except Exception as e:
                        await ui.display_error("An error occurred while processing your message.")
                        logger.exception("Error processing message")

        except Exception as e:
            logger.error("Error in terminal bot", error=str(e))
        finally:
            self._state = BotState.CLOSED
            await self._cleanup()

    @property
    def state(self) -> BotState:
        """Get the current bot state.

        Returns:
            BotState: Current state of the bot
        """
        return self._state

    async def __aenter__(self) -> ThreadSafeTerminalBot:
        """Enter async context.

        Returns:
            ThreadSafeTerminalBot: This instance
        """
        self._state = BotState.RUNNING
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self._state = BotState.CLOSED
        await self._cleanup()


async def stream_terminal_bot(
    graph: CompiledStateGraph = memgraph,
    user_input: dict[str, list[BaseMessage]] | None = None,
    config: RunnableConfig | None = None,
    interruptable: bool = False
) -> AsyncGenerator[str, None]:
    """Stream bot responses in the terminal.

    Args:
        graph: The graph to use for processing
        user_input: User input dictionary
        config: Optional runnable configuration
        interruptable: Whether the stream can be interrupted

    Yields:
        Bot response chunks

    Raises:
        RuntimeError: If memory limit is exceeded or task limit is reached
        ValueError: If no response is generated
    """
    bot = ThreadSafeTerminalBot()
    async with bot:
        async for chunk in bot.stream_terminal_bot(
            str(user_input),
            graph,
            config,
            interruptable
        ):
            yield chunk


async def invoke_terminal_bot(
    graph: CompiledStateGraph = memgraph,
    user_input: dict[str, list[BaseMessage]] | None = None,
    config: RunnableConfig | None = None
) -> str:
    """Invoke the terminal bot with input.

    Args:
        graph: The graph to use for processing
        user_input: User input dictionary
        config: Optional runnable configuration

    Returns:
        Bot response

    Raises:
        RuntimeError: If memory limit is exceeded or task limit is reached
        ValueError: If no response is generated
    """
    bot = ThreadSafeTerminalBot()
    async with bot:
        response, _ = await bot.invoke_terminal_bot(
            str(user_input),
            graph,
            config
        )
        return response


async def go_terminal_bot(graph: CompiledStateGraph = memgraph) -> None:
    """Start the terminal bot and run until shutdown.

    Args:
        graph: The graph to use for processing
    """
    bot = ThreadSafeTerminalBot()
    async with bot:
        await bot.start()


if __name__ == "__main__":
    asyncio.run(go_terminal_bot())
