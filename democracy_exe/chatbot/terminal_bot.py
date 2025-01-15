"""Terminal bot implementation for the chatbot."""
from __future__ import annotations


__all__ = ['ThreadSafeTerminalBot', 'stream_terminal_bot', 'invoke_terminal_bot', 'go_terminal_bot']

import asyncio
import signal
import sys
import threading

from collections.abc import AsyncGenerator
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.utils.resource_manager import ResourceLimits, ResourceManager


class BotState(Enum):
    """Enumeration of possible bot states."""
    RUNNING = auto()
    CLOSED = auto()


logger = structlog.get_logger(__name__)


class ThreadSafeTerminalBot:
    """Thread-safe terminal bot implementation."""

    def __init__(self) -> None:
        """Initialize the terminal bot."""
        # Get resource limits from settings or use defaults
        limits = ResourceLimits(
            max_memory_mb=getattr(aiosettings, "max_memory_mb", 512),
            max_tasks=getattr(aiosettings, "max_tasks", 100),
            max_response_size_mb=getattr(aiosettings, "max_response_size_mb", 1),
            max_buffer_size_kb=getattr(aiosettings, "max_buffer_size_kb", 64),
            task_timeout_seconds=getattr(aiosettings, "task_timeout_seconds", 30)
        )
        self._resource_manager = ResourceManager(limits=limits)
        self._shutdown_event = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        self._state = BotState.CLOSED
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
        stream_handler: Any | None = None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Stream bot responses in the terminal.

        Args:
            prompt: User input prompt
            stream_handler: Optional stream handler
            **kwargs: Additional keyword arguments

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
            buffer = []
            buffer_size = 0
            max_buffer = self._resource_manager.limits.max_buffer_size_kb * 1024

            async for chunk in self._stream_response(prompt, stream_handler, **kwargs):
                # Track memory for chunk
                chunk_size = len(chunk.encode('utf-8'))
                await self._resource_manager.track_memory(chunk_size)

                # Manage buffer
                buffer.append(chunk)
                buffer_size += chunk_size

                # Flush buffer if needed
                if buffer_size >= max_buffer:
                    combined = ''.join(buffer)
                    buffer = []
                    buffer_size = 0
                    yield combined

                # Release memory for processed chunk
                await self._resource_manager.release_memory(chunk_size)

            # Yield remaining buffer
            if buffer:
                yield ''.join(buffer)

        except Exception as e:
            logger.error("Error streaming response", error=str(e))
            raise
        finally:
            if task and task in self._tasks:
                self._tasks.remove(task)

    async def _stream_response(
        self,
        prompt: str,
        stream_handler: Any | None,
        **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        """Internal method to stream responses.

        Args:
            prompt: User input prompt
            stream_handler: Optional stream handler
            **kwargs: Additional keyword arguments

        Yields:
            Response chunks
        """
        if stream_handler:
            async for chunk in stream_handler(prompt, **kwargs):
                yield chunk
        else:
            yield prompt

    async def invoke_terminal_bot(
        self,
        prompt: str,
        **kwargs: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        """Invoke the terminal bot with a prompt.

        Args:
            prompt: User input prompt
            **kwargs: Additional keyword arguments

        Returns:
            Tuple of final answer and intermediate steps

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
            response_chunks = []
            total_size = 0
            max_size = self._resource_manager.limits.max_response_size_mb * 1024 * 1024

            async for chunk in self.stream_terminal_bot(prompt, **kwargs):
                chunk_size = len(chunk.encode('utf-8'))
                total_size += chunk_size

                if total_size > max_size:
                    raise RuntimeError(f"Response size exceeds limit of {max_size} bytes")

                response_chunks.append(chunk)

            response = ''.join(response_chunks)
            if not response:
                raise ValueError("No response generated")

            return response, []  # Empty list for intermediate steps

        except Exception as e:
            logger.error("Error invoking bot", error=str(e))
            raise
        finally:
            if task and task in self._tasks:
                self._tasks.remove(task)

    @property
    def state(self) -> BotState:
        """Get the current bot state.

        Returns:
            BotState: Current state of the bot
        """
        return self._state

    async def start(self) -> None:
        """Start the terminal bot."""
        try:
            logger.info("Starting terminal bot")
            self._state = BotState.RUNNING
            await self._shutdown_event.wait()
        except Exception as e:
            logger.error("Error in terminal bot", error=str(e))
        finally:
            self._state = BotState.CLOSED
            await self._cleanup()

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
    graph: Any,
    user_input: dict[str, Any],
    stream_handler: Any | None = None
) -> AsyncGenerator[str, None]:
    """Stream bot responses in the terminal.

    Args:
        graph: The graph to use for processing
        user_input: User input dictionary
        stream_handler: Optional stream handler

    Yields:
        Bot response chunks

    Raises:
        RuntimeError: If memory limit is exceeded or task limit is reached
        ValueError: If no response is generated
    """
    bot = ThreadSafeTerminalBot()
    async with bot:
        async for chunk in bot.stream_terminal_bot(str(user_input), stream_handler):
            yield chunk


async def invoke_terminal_bot(
    graph: Any,
    user_input: dict[str, Any],
    handler: Any | None = None
) -> str:
    """Invoke the terminal bot with input.

    Args:
        graph: The graph to use for processing
        user_input: User input dictionary
        handler: Optional message handler

    Returns:
        Bot response

    Raises:
        RuntimeError: If memory limit is exceeded or task limit is reached
        ValueError: If no response is generated
    """
    bot = ThreadSafeTerminalBot()
    async with bot:
        response, _ = await bot.invoke_terminal_bot(str(user_input))
        return response


async def go_terminal_bot() -> None:
    """Start the terminal bot and run until shutdown."""
    bot = ThreadSafeTerminalBot()
    async with bot:
        await bot.start()
