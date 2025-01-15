# """Extension loading utilities for the Discord bot.

# This module contains functions for managing Discord bot extension loading,
# including dependency resolution and retry logic.
# """
# from __future__ import annotations

# import asyncio

# from typing import TYPE_CHECKING

# import structlog

# from discord.ext import commands


# logger = structlog.get_logger(__name__)


# if TYPE_CHECKING:
#     from democracy_exe.chatbot.core.bot import DemocracyBot


# def get_extension_load_order(extensions: list[str]) -> list[str]:
#     """Get the extension load order based on dependencies.

#     Args:
#         extensions: List of extensions to order

#     Returns:
#         List of extension names in dependency order

#     Raises:
#         ValueError: If circular dependencies are detected
#     """
#     # Define extension dependencies (extension -> list of dependencies)
#     dependencies = {
#         'democracy_exe.chatbot.cogs.core': [],
#         'democracy_exe.chatbot.cogs.admin': ['democracy_exe.chatbot.cogs.core'],
#         'democracy_exe.chatbot.cogs.ai': ['democracy_exe.chatbot.cogs.core'],
#         'democracy_exe.chatbot.cogs.utils': ['democracy_exe.chatbot.cogs.core'],
#     }

#     # Topological sort
#     load_order = []
#     visited = set()
#     temp_visited = set()

#     def visit(extension: str) -> None:
#         if extension in temp_visited:
#             raise ValueError(f"Circular dependency detected for {extension}")
#         if extension not in visited:
#             temp_visited.add(extension)
#             for dep in dependencies.get(extension, []):
#                 visit(dep)
#             temp_visited.remove(extension)
#             visited.add(extension)
#             load_order.append(extension)

#     for extension in extensions:
#         if extension not in visited:
#             visit(extension)

#     return load_order


# async def load_extension_with_retry(bot: DemocracyBot, extension: str, max_retries: int) -> None:
#     """Load an extension with exponential backoff retry.

#     Args:
#         bot: The bot instance to load the extension into
#         extension: The extension to load
#         max_retries: Maximum number of retry attempts

#     Raises:
#         RuntimeError: If extension fails to load after retries
#     """
#     base_delay = 1.0
#     for attempt in range(max_retries):
#         try:
#             await bot.load_extension(extension)
#             logger.info("Loaded extension", extension=extension)
#             break
#         except Exception as e:
#             if attempt == max_retries - 1:
#                 logger.error("Failed to load extension",
#                            extension=extension,
#                            error=str(e))
#                 raise RuntimeError(f"Failed to load extension {extension} after {max_retries} attempts")
#             delay = base_delay * (2 ** attempt)  # Exponential backoff
#             logger.warning("Extension load failed, retrying",
#                          extension=extension,
#                          attempt=attempt + 1,
#                          delay=delay)
#             await asyncio.sleep(delay)
