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


@pytest.fixture
async def bot() -> DemocracyBot:
    """Create a DemocracyBot instance for testing.

    Returns:
        DemocracyBot: The bot instance for testing
    """
    intents = discord.Intents.default()
    intents.message_content = True
    intents.guilds = True
    bot = DemocracyBot(command_prefix="?", intents=intents)
    yield bot
    if not bot.is_closed():
        await bot.close()
    if hasattr(bot, "session") and not bot.session.closed:
        await bot.session.close()


@pytest.fixture
def mock_cogs_directory(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a mock cogs directory with test files.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to mock cogs directory
    """
    cogs_dir = tmp_path / "chatbot" / "cogs"
    cogs_dir.mkdir(parents=True)

    # Create test cog files
    (cogs_dir / "test_cog1.py").write_text("# Test cog 1")
    (cogs_dir / "test_cog2.py").write_text("# Test cog 2")
    (cogs_dir / "__init__.py").write_text("")

    # Create subdirectory with another cog
    subcategory = cogs_dir / "subcategory"
    subcategory.mkdir()
    (subcategory / "test_cog3.py").write_text("# Test cog 3")
    (subcategory / "__init__.py").write_text("")

    return cogs_dir
