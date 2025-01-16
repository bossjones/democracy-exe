# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Terminal UI management utilities for the chatbot.

This module provides utilities for managing the terminal user interface.
It includes functionality for:
- User input handling
- Terminal output formatting
- Status and error display
- Progress indicators

Key Components:
    - UIManager: Core class for UI management
    - InputHandler: Manages user input processing
    - OutputFormatter: Handles output formatting

Dependencies:
    - rich: For terminal formatting
    - structlog: For logging
    - asyncio: For async IO operations

Example:
    ```python
    ui = UIManager()
    async with ui.managed_io():
        user_input = await ui.get_input("You: ")
        await ui.display_response("Processing...")
    ```

Note:
    This module is designed to provide a clean and responsive terminal interface
    while maintaining proper async operation and error handling.
"""
from __future__ import annotations

import asyncio
import sys

from collections.abc import AsyncGenerator
from typing import Any, Optional

import structlog

from rich import print as rprint


logger = structlog.get_logger(__name__)


class FlushingStderr:
    """A class to handle flushing stderr output."""

    def write(self, message: str) -> None:
        """Write and flush a message to stderr.

        Args:
            message: The message to write to stderr
        """
        sys.stderr.write(message)
        sys.stderr.flush()


class UIManager:
    """Manages terminal UI interactions."""

    def __init__(self) -> None:
        """Initialize the UI manager."""
        self._logger = logger.bind(module="UIManager")
        self._stderr_handler = FlushingStderr()

    async def get_input(self, prompt: str = "You: ") -> str:
        """Get user input asynchronously.

        Args:
            prompt: Input prompt to display

        Returns:
            str: User input
        """
        try:
            # Flush stderr before input prompt
            sys.stderr.flush()
            return await asyncio.to_thread(input, prompt)
        except Exception as e:
            self._logger.error("Error getting input", error=str(e))
            raise

    async def display_welcome(self) -> None:
        """Display welcome message."""
        rprint("[bold green]Welcome to the DemocracyExeAI Chatbot! Type 'quit' to exit.[/bold green]")
        self._logger.info("Welcome to the DemocracyExeAI Chatbot! Type 'quit' to exit.")

    async def display_goodbye(self) -> None:
        """Display goodbye message."""
        rprint("[bold red]Goodbye![/bold red]")
        self._logger.info("Goodbye!")

    async def display_error(self, message: str) -> None:
        """Display error message.

        Args:
            message: Error message to display
        """
        rprint(f"[bold red]{message}[/bold red]")
        self._logger.error(message)

    async def display_response(self, message: str) -> None:
        """Display bot response.

        Args:
            message: Response message to display
        """
        rprint(f"[bold blue]AI:[/bold blue] {message}")
        self._logger.info(f"AI: {message}")

    async def __aenter__(self) -> UIManager:
        """Enter async context.

        Returns:
            UIManager: This instance
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        # Ensure stderr is flushed
        sys.stderr.flush()
