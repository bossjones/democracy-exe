# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Type definitions for the chatbot core."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from discord.ext import commands
    from redis.asyncio import ConnectionPool as RedisConnectionPool

class BotProtocol(Protocol):
    """Protocol defining the interface for the DemocracyBot."""

    pool: RedisConnectionPool | None
    command_prefix: str | list[str]
