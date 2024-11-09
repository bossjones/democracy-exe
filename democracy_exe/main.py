"""Entry point for the sandbox agent."""

from __future__ import annotations

import asyncio

from loguru import logger as LOGGER

from democracy_exe.aio_settings import aiosettings
from democracy_exe.bot import SandboxAgent


async def main():
    async with SandboxAgent() as bot:
        # if aiosettings.enable_redis:
        #     bot.pool = pool
        await bot.start()

    await LOGGER.complete()


if __name__ == "__main__":
    asyncio.run(main())
