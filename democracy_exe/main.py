"""Entry point for the sandbox agent."""

from __future__ import annotations

import asyncio

import structlog


logger = structlog.get_logger(__name__)

from democracy_exe.aio_settings import aiosettings


async def main():
    logger.info("Starting sandbox agent")



if __name__ == "__main__":
    asyncio.run(main())
