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
