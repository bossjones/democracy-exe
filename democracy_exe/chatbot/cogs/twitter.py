"""Twitter cog for Discord bot.

This cog provides Twitter-related functionality including downloading tweets,
threads, cards and displaying tweet metadata.

Attributes:
    bot: The Discord bot instance
"""
from __future__ import annotations

import asyncio
import pathlib
import tempfile

from typing import Any, Dict, Final, List, Optional, Tuple, Union, cast

import discord

from discord.ext import commands
from discord.ext.commands import Context
from loguru import logger
from rich.pretty import pprint

from democracy_exe.utils.twitter_utils.download import download_tweet
from democracy_exe.utils.twitter_utils.embed import (
    TweetMetadata,
    create_card_embed,
    create_download_progress_embed,
    create_error_embed,
    create_info_embed,
    create_thread_embed,
    create_tweet_embed,
)
from democracy_exe.utils.twitter_utils.types import DownloadResult, TweetDownloadMode


# Command constants
HELP_MESSAGE: Final[str] = """
Available commands:
- !tweet download <url>: Download tweet media and metadata
- !tweet thread <url>: Download full tweet thread
- !tweet card <url>: Download tweet card preview
- !tweet info <url>: Show tweet metadata
"""


class TwitterError(Exception):
    """Base exception for Twitter-related errors."""


class Twitter(commands.Cog):
    """Twitter functionality for Discord bot.

    Handles downloading and displaying tweets, threads and cards.

    Attributes:
        bot: The Discord bot instance
    """

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    async def _handle_download(
        self,
        ctx: Context,
        url: str,
        mode: TweetDownloadMode
    ) -> tuple[bool, str | None]:
        """Handle tweet download and send response.

        Args:
            ctx: Command context
            url: Tweet URL
            mode: Download mode (single/thread/card)

        Returns:
            Tuple containing:
                - Success status (bool)
                - Error message if any (Optional[str])

        Raises:
            TwitterError: If download fails
        """
        """Handle tweet download and send response.

        Args:
            ctx: Command context
            url: Tweet URL
            mode: Download mode (single/thread/card)

        Raises:
            TwitterError: If download fails
        """
        async with ctx.typing():
            # Create progress message with embed
            progress_embed = create_download_progress_embed(url, mode)
            progress = await ctx.send(embed=progress_embed)

            try:
                # Download tweet content
                result = await download_tweet(url, mode=mode)

                if not result["success"]:
                    error_embed = create_error_embed(result["error"])
                    await progress.edit(embed=error_embed)
                    return

                # Create appropriate embed based on mode
                if mode == "thread":
                    metadata_list = cast(list[TweetMetadata], result["metadata"])
                    embed = create_thread_embed(metadata_list)
                elif mode == "card":
                    metadata = cast(TweetMetadata, result["metadata"])
                    embed = create_card_embed(metadata)
                else:
                    metadata = cast(TweetMetadata, result["metadata"])
                    embed = create_tweet_embed(metadata)

                # Upload media files if any
                files = []
                for file_path in result["local_files"]:
                    try:
                        files.append(discord.File(file_path))
                    except Exception as e:
                        logger.warning(f"Failed to create discord.File for {file_path}: {e}")

                await progress.edit(content="Download complete!", embed=embed)
                if files:
                    await ctx.send(files=files)
                return True, None

            except Exception as e:
                logger.exception("Error downloading tweet")
                error_embed = create_error_embed(str(e))
                await progress.edit(embed=error_embed)
                error_msg = f"Failed to download tweet: {e}"
                raise TwitterError(error_msg) from e

    @commands.group(name="tweet")
    async def tweet(self, ctx: commands.Context) -> None:
        """Twitter command group."""
        if ctx.invoked_subcommand is None:
            await ctx.send(HELP_MESSAGE)

    @tweet.command(name="download")
    async def download(self, ctx: commands.Context, url: str) -> None:
        """Download tweet media and metadata.

        Args:
            ctx: Command context
            url: Tweet URL to download
        """
        await self._handle_download(ctx, url, mode="single")

    @tweet.command(name="thread")
    async def thread(self, ctx: commands.Context, url: str) -> None:
        """Download full tweet thread.

        Args:
            ctx: Command context
            url: Tweet thread URL to download
        """
        await self._handle_download(ctx, url, mode="thread")

    @tweet.command(name="card")
    async def card(self, ctx: commands.Context, url: str) -> None:
        """Download tweet card preview.

        Args:
            ctx: Command context
            url: Tweet URL to download card from
        """
        await self._handle_download(ctx, url, mode="card")

    @tweet.command(name="info")
    async def info(self, ctx: commands.Context, url: str) -> None:
        """Show tweet metadata.

        Args:
            ctx: Command context
            url: Tweet URL to get info for

        Raises:
            TwitterError: If getting info fails
        """
        async with ctx.typing():
            try:
                result = await download_tweet(url, mode="single")

                if not result["success"]:
                    error_embed = create_error_embed(result["error"])
                    await ctx.send(embed=error_embed)
                    return

                # Create detailed info embed
                metadata = cast(TweetMetadata, result["metadata"])
                embed = create_info_embed(metadata)
                await ctx.send(embed=embed)

            except Exception as e:
                logger.exception("Error getting tweet info")
                error_embed = create_error_embed(str(e))
                await ctx.send(embed=error_embed)
                raise TwitterError(f"Failed to get tweet info: {e}") from e


async def setup(bot: commands.Bot) -> None:
    """Add Twitter cog to bot.

    Args:
        bot: Discord bot instance
    """
    await bot.add_cog(Twitter(bot))
