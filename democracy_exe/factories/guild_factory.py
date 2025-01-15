"""Guild factory for creating guild instances."""
from __future__ import annotations

import asyncio
import os

from typing import Any, Dict, Optional

import structlog

from democracy_exe.aio_settings import aiosettings
from democracy_exe.factories.singleton import Singleton


logger = structlog.get_logger(__name__)


class Guild(metaclass=Singleton):
    """Guild class for managing Discord guild data."""

    def __init__(self, id: int | None = None, prefix: str | None = None) -> None:
        """Initialize the guild.

        Args:
            id: Guild ID
            prefix: Command prefix
        """
        self.id = id if id is not None else getattr(aiosettings, "discord_server_id", 0)
        self.prefix = prefix if prefix is not None else getattr(aiosettings, "prefix", "?")
        self._lock = asyncio.Lock()
        self._data: dict[str, Any] = {}

    async def get_data(self, key: str) -> Any:
        """Get guild data by key.

        Args:
            key: Data key

        Returns:
            Any: Data value
        """
        async with self._lock:
            return self._data.get(key)

    async def set_data(self, key: str, value: Any) -> None:
        """Set guild data by key.

        Args:
            key: Data key
            value: Data value
        """
        async with self._lock:
            self._data[key] = value

    async def delete_data(self, key: str) -> None:
        """Delete guild data by key.

        Args:
            key: Data key
        """
        async with self._lock:
            self._data.pop(key, None)

    async def clear_data(self) -> None:
        """Clear all guild data."""
        async with self._lock:
            self._data.clear()

    @property
    def data(self) -> dict[str, Any]:
        """Get all guild data.

        Returns:
            Dict[str, Any]: Guild data
        """
        return self._data.copy()

    def __str__(self) -> str:
        """Get string representation.

        Returns:
            str: String representation
        """
        return f"Guild(id={self.id}, prefix={self.prefix})"


# smoke tests
if __name__ == "__main__":
    test_guild_metadata = Guild(id=int(aiosettings.discord_server_id), prefix=aiosettings.prefix)
    print(test_guild_metadata)
    print(test_guild_metadata.id)
    print(test_guild_metadata.prefix)

    test_guild_metadata2 = Guild()
    print(test_guild_metadata2)
    print(test_guild_metadata2.id)
    print(test_guild_metadata2.prefix)
