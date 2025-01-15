# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Bot-specific context implementation.

This module contains the bot-specific context class that extends the base context.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import discord

from discord.ext import commands

from democracy_exe.chatbot.core.types import BotProtocol
from democracy_exe.utils.base_context import BaseContext


if TYPE_CHECKING:
    from redis.asyncio import ConnectionPool as RedisConnectionPool

class Context(BaseContext):
    """Bot-specific context class for Discord commands."""

    bot: BotProtocol

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the bot context.

        Args:
            **kwargs: Keyword arguments to pass to the parent class
        """
        super().__init__(**kwargs)
        self.pool: RedisConnectionPool | None = self.bot.pool
