from __future__ import annotations

import asyncio
import signal
import sys
import threading
import weakref

from collections.abc import AsyncGenerator, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union, cast

import structlog

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from rich import print as rprint

from democracy_exe.agentic import _utils as agentic_utils
from democracy_exe.agentic.workflows.react.graph import graph as memgraph
from democracy_exe.aio_settings import aiosettings


logger = structlog.get_logger(__name__)


class BotState:
    """Enumeration of possible bot states."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    CLOSED = "closed"


class TerminalBotState(TypedDict):
    """Type definition for terminal bot state."""

    messages: list[BaseMessage]
    response: str | None


class ThreadSafeTerminalBot:
    """Thread-safe terminal bot implementation with proper resource management."""

    def __init__(self) -> None:
        """Initialize the terminal bot with thread safety mechanisms."""
        self._lock = asyncio.Lock()
        self._cleanup_lock = asyncio.Lock()
        self._closed = False
        self._sem = asyncio.Semaphore(1)
        self._state = BotState.INITIALIZING
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._tasks: weakref.WeakSet = weakref.WeakSet()
        self._loop: asyncio.AbstractEventLoop | None = None

        # Set up signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle system signals for graceful shutdown.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._initiate_shutdown)

    async def _initiate_shutdown(self) -> None:
        """Initiate graceful shutdown sequence."""
        if self._state != BotState.SHUTTING_DOWN:
            self._state = BotState.SHUTTING_DOWN
            await self.cleanup()

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get the event loop for the current context.

        Returns:
            The event loop to use

        Raises:
            RuntimeError: If called from wrong thread
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    async def __aenter__(self) -> ThreadSafeTerminalBot:
        """Enter the async context.

        Returns:
            ThreadSafeTerminalBot: The bot instance

        Raises:
            RuntimeError: If bot is already closed
        """
        if self._closed:
            raise RuntimeError("Bot is closed")

        self._loop = self._get_loop()
        self._state = BotState.RUNNING
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any | None) -> None:
        """Exit the async context with proper cleanup.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        await self._initiate_shutdown()

    async def cleanup(self) -> None:
        """Clean up resources safely."""
        if not self._closed:
            async with self._cleanup_lock:
                try:
                    # Cancel all pending tasks
                    for task in self._tasks:
                        if not task.done():
                            task.cancel()

                    # Wait for tasks to complete
                    with suppress(asyncio.CancelledError):
                        pending = [t for t in self._tasks if not t.done()]
                        if pending:
                            await asyncio.gather(*pending, return_exceptions=True)

                    # Shutdown thread pool
                    self._executor.shutdown(wait=True)

                    # Clear event loop reference
                    if self._loop is not None:
                        self._loop = None

                finally:
                    self._closed = True
                    self._state = BotState.CLOSED

    async def process_message(
        self,
        graph: CompiledStateGraph,
        message: str,
        config: RunnableConfig
    ) -> None:
        """Process a user message safely.

        Args:
            graph: The graph to process the message with
            message: The user's message
            config: Configuration for processing

        Raises:
            RuntimeError: If bot is not in running state
        """
        if self._state != BotState.RUNNING:
            raise RuntimeError(f"Bot is in {self._state} state")

        async with self._sem:
            try:
                human_message = HumanMessage(content=message)
                user_input_dict: dict[str, list[BaseMessage]] = {"messages": [human_message]}
                stream_terminal_bot(graph, user_input_dict, config)
            except Exception as e:
                logger.exception("Error processing message", error=str(e))
                raise


class FlushingStderr:
    """A class to handle flushing stderr output."""

    def write(self, message: str) -> None:
        """Write and flush a message to stderr.

        Args:
            message: The message to write to stderr
        """
        sys.stderr.write(message)
        sys.stderr.flush()


async def go_terminal_bot(graph: CompiledStateGraph = memgraph) -> None:
    """Main function to run the LangGraph Chatbot in the terminal.

    This function handles user input and processes it through the AI pipeline.
    It ensures proper cleanup of resources and handles errors gracefully.

    Args:
        graph: The compiled state graph to use for processing messages
    """
    logger.info("Starting the DemocracyExeAI Chatbot")
    rprint("[bold green]Welcome to the DemocracyExeAI Chatbot! Type 'quit' to exit.[/bold green]")
    logger.info("Welcome to the DemocracyExeAI Chatbot! Type 'quit' to exit.")

    config: RunnableConfig = {"configurable": {"thread_id": "1", "user_id": "1"}}

    try:
        async with ThreadSafeTerminalBot() as bot:
            while True:
                try:
                    user_input = await asyncio.to_thread(input, "You: ")

                    if user_input.lower() == 'quit':
                        rprint("[bold red]Goodbye![/bold red]")
                        logger.info("Goodbye!")
                        break

                    await bot.process_message(graph, user_input, config)
                except asyncio.CancelledError:
                    logger.info("Bot operation cancelled")
                    break
                except Exception as e:
                    logger.exception("Error in main bot loop", error=str(e))
                    continue
    except Exception as e:
        logger.exception("Fatal error in bot", error=str(e))
        raise
    finally:
        # Ensure event loop is cleaned up
        try:
            loop = asyncio.get_running_loop()
            tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks, return_exceptions=True)
            loop.stop()
        except Exception as e:
            logger.exception("Error cleaning up event loop", error=str(e))


def stream_terminal_bot(
    graph: CompiledStateGraph = memgraph,
    user_input: dict[str, list[BaseMessage]] | None = None,
    thread: RunnableConfig | None = None,
    interruptable: bool = False
) -> None:
    """Stream the LangGraph Chatbot in the terminal.

    This function processes user input through the graph and streams the responses.
    It handles interruptions and user approval for tool calls.

    Args:
        graph: The compiled state graph to use for processing messages
        user_input: Dictionary containing user messages
        thread: Thread configuration dictionary
        interruptable: Whether the stream can be interrupted for user approval

    Raises:
        Exception: If an error occurs during streaming
    """
    try:
        # Run the graph until the first interruption
        for event in graph.stream(user_input, thread, stream_mode="values"):
            logger.debug("Processing event", event=event)
            chunk = event['messages'][-1]
            logger.debug("Processing chunk", chunk=chunk, type=type(chunk))

            chunk.pretty_print()

        if interruptable:
            # Get user feedback
            user_approval = input("Do you want to call the tool? (yes[y]/no[n]): ")

            # Check approval
            if user_approval.lower() in ("yes", "y"):
                # If approved, continue the graph execution
                for event in graph.stream(None, thread, stream_mode="values"):
                    event['messages'][-1].pretty_print()
            else:
                print("Operation cancelled by user.")
    except Exception as e:
        logger.exception("Error in stream processing", error=str(e))
        raise


def invoke_terminal_bot(
    graph: CompiledStateGraph = memgraph,
    user_input: dict[str, list[BaseMessage]] | None = None,
    thread: RunnableConfig | None = None
) -> str | None:
    """Invoke the LangGraph Chatbot in the terminal.

    This function processes a single message through the graph and returns the response.

    Args:
        graph: The compiled state graph to use for processing messages
        user_input: Dictionary containing user messages
        thread: Thread configuration dictionary

    Returns:
        The AI response message as a string, or None if no response

    Raises:
        Exception: If an error occurs during processing
    """
    try:
        messages = graph.invoke(user_input, thread)
        for m in messages['messages']:
            m.pretty_print()
        response = cast(str, messages['response'])
        rprint(f"[bold blue]AI:[/bold blue] {response}")
        logger.info("AI response", response=response)
        return response
    except Exception as e:
        logger.exception("Error invoking bot", error=str(e))
        raise


if __name__ == "__main__":
    try:
        asyncio.run(go_terminal_bot())
    except KeyboardInterrupt:
        logger.info("Bot terminated by user")
    except Exception as e:
        logger.exception("Bot terminated with error", error=str(e))
        sys.exit(1)
