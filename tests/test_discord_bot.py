# FIXME: dlete this file
"""Tests for Discord bot functionality.

This module contains tests for the Discord bot's core functionality.
"""

from __future__ import annotations

import os
import pathlib
import tempfile

from typing import TYPE_CHECKING, Any, List

import discord

from structlog.testing import capture_logs

import pytest

from democracy_exe.chatbot.core.bot import DemocracyBot
from democracy_exe.chatbot.utils.discord_utils import extensions


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


# # FIXME: compare this to the one below 1/20/2025
# @pytest_asyncio.fixture
# async def bot() -> AsyncGenerator[DemocracyBot, None]:
#     """Create a DemocracyBot instance for testing.

#     Args:
#         event_loop: The event loop fixture

#     Returns:
#         AsyncGenerator[DemocracyBot, None]: DemocracyBot instance with test configuration
#     """
#     # Configure intents
#     intents = discord.Intents.default()
#     intents.members = True
#     intents.message_content = True
#     intents.messages = True
#     intents.guilds = True

#     # Create DemocracyBot with test configuration
#     bot = DemocracyBot(command_prefix="?", intents=intents, description="Test DemocracyBot instance")

#     # set up the loop
#     if isinstance(bot.loop, _LoopSentinel):  # type: ignore
#         await bot._async_setup_hook()  # type: ignore

#     # Add test-specific error handling
#     @bot.event
#     async def on_command_error(ctx: commands.Context, error: Exception) -> None:  # type: ignore
#         """Handle command errors in test environment."""
#         raise error  # Re-raise for pytest to catch

#     # Setup and cleanup
#     # await bot._async_setup_hook()  # Required for proper initialization
#     # await dpytest.empty_queue()
#     dpytest.configure(bot)
#     yield bot
#     # await dpytest.empty_queue()

#     try:
#         # Teardown
#         await dpytest.empty_queue()  # empty the global message queue as test teardown
#     finally:
#         pass


# @pytest.fixture
# async def bot() -> DemocracyBot:
#     """Create a DemocracyBot instance for testing.

#     Returns:
#         DemocracyBot: The bot instance for testing
#     """
#     intents = discord.Intents.default()
#     intents.message_content = True
#     intents.guilds = True
#     bot = DemocracyBot(command_prefix="?", intents=intents)
#     yield bot
#     if not bot.is_closed():
#         await bot.close()
#     if hasattr(bot, "session") and not bot.session.closed:
#         await bot.session.close()


# @pytest.fixture
# def mock_cogs_directory(tmp_path: pathlib.Path) -> pathlib.Path:
#     """Create a mock cogs directory with test files.

#     Args:
#         tmp_path: Pytest temporary directory fixture

#     Returns:
#         Path to mock cogs directory
#     """
#     cogs_dir = tmp_path / "chatbot" / "cogs"
#     cogs_dir.mkdir(parents=True)

#     # Create test cog files
#     (cogs_dir / "test_cog1.py").write_text("# Test cog 1")
#     (cogs_dir / "test_cog2.py").write_text("# Test cog 2")
#     (cogs_dir / "__init__.py").write_text("")

#     # Create subdirectory with another cog
#     subcategory = cogs_dir / "subcategory"
#     subcategory.mkdir()
#     (subcategory / "test_cog3.py").write_text("# Test cog 3")
#     (subcategory / "__init__.py").write_text("")

#     return cogs_dir
