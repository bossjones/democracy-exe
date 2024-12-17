# pyright: reportAttributeAccessIssue=false

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

from democracy_exe.factories.guild_factory import Guild
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
        logger.debug("Initializing Twitter cog")
        self.bot = bot
        logger.debug("Twitter cog initialized")


    @commands.Cog.listener()
    async def on_ready(self):
        logger.debug(f"{type(self).__name__} Cog ready.")
        print(f"{type(self).__name__} Cog ready.")
        await logger.complete()

    @commands.Cog.listener()
    async def on_guild_join(self, guild):
        """Add new guilds to the database"""
        logger.debug(f"Adding new guild to database: {guild.id}")
        guild_obj = Guild(id=guild.id)
        logger.debug(f"Successfully added guild {guild.id} to database")

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
        logger.debug(f"Starting download handler - URL: {url}, Mode: {mode}")
        async with ctx.typing():
            # Create progress message with embed
            progress_embed = create_download_progress_embed(url, mode)
            progress = await ctx.send(embed=progress_embed)
            logger.debug("Created progress embed")

            try:
                # Download tweet content
                logger.debug("Initiating tweet download")
                result = await download_tweet(url, mode=mode)
                logger.debug(f"Download result: success={result['success']}")

                if not result["success"]:
                    logger.debug(f"Download failed: {result['error']}")
                    error_embed = create_error_embed(str(result.get("error", "Unknown error")))
                    await progress.edit(embed=error_embed)
                    return False, result["error"]

                # Create appropriate embed based on mode
                logger.debug("Creating response embed")
                if mode == "thread":
                    metadata_list = cast(list[TweetMetadata], result["metadata"])
                    embed = create_thread_embed(metadata_list)
                    logger.debug(f"Created thread embed with {len(metadata_list)} tweets")
                elif mode == "card":
                    metadata = cast(TweetMetadata, result["metadata"])
                    embed = create_card_embed(metadata)
                    logger.debug("Created card embed")
                else:
                    metadata = cast(TweetMetadata, result["metadata"])
                    embed = create_tweet_embed(metadata)
                    logger.debug("Created single tweet embed")

                # Upload media files if any
                files = []
                logger.debug(f"Processing {len(result['local_files'])} media files")
                for file_path in result["local_files"]:
                    try:
                        files.append(discord.File(file_path))
                        logger.debug(f"Added file to upload: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create discord.File for {file_path}: {e}")

                await progress.edit(content="Download complete!", embed=embed)
                if files:
                    logger.debug(f"Uploading {len(files)} media files")
                    await ctx.send(files=files)
                logger.debug("Download handler completed successfully")
                return True, None

            except Exception as e:
                logger.exception("Error in download handler")
                error_embed = create_error_embed(str(e))
                await progress.edit(embed=error_embed)
                error_msg = f"Failed to download tweet: {e}"
                raise TwitterError(error_msg) from e

    @commands.group(name="tweet")
    async def tweet(self, ctx: commands.Context) -> None:
        """Twitter command group."""
        logger.debug(f"Tweet command invoked by {ctx.author} in {ctx.guild}")
        if ctx.invoked_subcommand is None: # type: ignore
            logger.debug("No subcommand specified, sending help message")
            await ctx.send(HELP_MESSAGE)

    @tweet.command(name="download", aliases=["dlt", "t", "twitter"])
    async def download(self, ctx: commands.Context, url: str) -> None:
        """Download tweet media and metadata.

        Args:
            ctx: Command context
            url: Tweet URL to download
        """
        logger.debug(f"Download command invoked - URL: {url}")
        await self._handle_download(ctx, url, mode="single")
        logger.debug("Download command completed")

    @tweet.command(name="thread", aliases=["dt"])
    async def thread(self, ctx: commands.Context, url: str) -> None:
        """Download full tweet thread.

        Args:
            ctx: Command context
            url: Tweet thread URL to download
        """
        logger.debug(f"Thread command invoked - URL: {url}")
        await self._handle_download(ctx, url, mode="thread")
        logger.debug("Thread command completed")

    @tweet.command(name="card")
    async def card(self, ctx: commands.Context, url: str) -> None:
        """Download tweet card preview.

        Args:
            ctx: Command context
            url: Tweet URL to download card from
        """
        logger.debug(f"Card command invoked - URL: {url}")
        await self._handle_download(ctx, url, mode="card")
        logger.debug("Card command completed")

    @tweet.command(name="info")
    async def info(self, ctx: commands.Context, url: str) -> None:
        """Show tweet metadata.

        Args:
            ctx: Command context
            url: Tweet URL to get info for

        Raises:
            TwitterError: If getting info fails
        """
        logger.debug(f"Info command invoked - URL: {url}")
        async with ctx.typing():
            try:
                logger.debug("Fetching tweet metadata")
                result = await download_tweet(url, mode="single")
                logger.debug(f"Metadata fetch result: success={result['success']}")

                if not result["success"]:
                    logger.debug(f"Metadata fetch failed: {result['error']}")
                    error_embed = create_error_embed(result["error"])
                    await ctx.send(embed=error_embed)
                    return

                # Create detailed info embed
                metadata = cast(TweetMetadata, result["metadata"])
                embed = create_info_embed(metadata)
                logger.debug("Created info embed")
                await ctx.send(embed=embed)
                logger.debug("Info command completed successfully")

            except Exception as e:
                logger.exception("Error in info command")
                error_embed = create_error_embed(str(e))
                await ctx.send(embed=error_embed)
                raise TwitterError(f"Failed to get tweet info: {e}") from e


async def setup(bot: commands.Bot) -> None:
    """Add Twitter cog to bot.

    Args:
        bot: Discord bot instance
    """
    logger.debug("Setting up Twitter cog")
    await bot.add_cog(Twitter(bot))
    logger.debug("Twitter cog setup complete")
