"""Entry point for the sandbox agent."""

from __future__ import annotations

import asyncio

# from loguru import logger
import structlog


logger = structlog.get_logger(__name__)

from democracy_exe.aio_settings import aiosettings


# from democracy_exe.bot import SandboxAgent


async def main():
    logger.info("Starting sandbox agent")
    # async with SandboxAgent() as bot:
    #     # if aiosettings.enable_redis:
    #     #     bot.pool = pool
    #     await bot.start()

    # await logger.complete()


if __name__ == "__main__":
    asyncio.run(main())
