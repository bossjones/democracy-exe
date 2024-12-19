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
import sys
import tempfile
import traceback

from typing import Any, Dict, Final, List, Optional, Tuple, Union, cast

import bpdb
import discord

from discord.ext import commands
from discord.ext.commands import Context
from loguru import logger
from rich.pretty import pprint

from democracy_exe.aio_settings import aiosettings
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
HELP_MESSAGE: Final[str] = f"""
Available commands:
- {aiosettings.prefix}tweet download <url>: Download tweet media and metadata
- {aiosettings.prefix}tweet thread <url>: Download full tweet thread
- {aiosettings.prefix}tweet card <url>: Download tweet card preview
- {aiosettings.prefix}tweet info <url>: Show tweet metadata
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

    def _cleanup_temp_dir(self, file_path: str) -> None:
        """Delete temporary directory created for tweet downloads.

        Args:
            file_path: Path to a file in the temporary directory

        Note:
            Silently fails if directory doesn't exist or can't be deleted to avoid
            disrupting the main download flow.
        """
        try:
            # Get the parent directory of the file
            temp_dir = pathlib.Path(file_path).parent
            logger.debug(f"temp_dir: {temp_dir}")

            # Verify that we're deleting a gallery-dl directory
            if temp_dir.exists():
                logger.debug(f"Cleaning up temporary directory: {temp_dir}")
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug("Temporary directory cleanup complete")
            else:
                logger.warning(f"Skipping cleanup - directory {temp_dir} doesn't exist")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory: {e}")

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
        logger.info(f"{type(self).__name__} -> _handle_download -> ctx = {ctx}, url = {url}, mode = {mode}")
        logger.debug(f"Starting download handler - URL: {url}, Mode: {mode}")
        # async with ctx.typing():
        # Create progress message with embed
        progress_embed = create_download_progress_embed(url, mode)
        progress = await ctx.send(embed=progress_embed)
        logger.debug("Created progress embed")

        try:
            # Download tweet content
            logger.debug("Initiating tweet download")
            result = await download_tweet(url, mode=mode)
            logger.error(f"result: {result}")
            logger.debug(f"Download result: success={result['success']}")

            if not result["success"]:
                logger.debug(f"Download failed: {result['error']}")
                error_embed = create_error_embed(str(result.get("error", "Unknown error")))
                await progress.edit(embed=error_embed)
                return False, result["error"]


            # Create appropriate embed based on mode
            logger.debug("Creating response embed")
            if mode == "thread":
                metadata_list = [cast(TweetMetadata, result["metadata"])]
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
            # import bpdb; bpdb.set_trace()
            # {
            # 'success': True,
            # 'metadata': {'id': '', 'url': '', 'author': '', 'content': '', 'media_urls': [], 'created_at': ''},
            # 'local_files': ['/private/var/folders/q_/d5r_s8wd02zdx6qmc5f_96mw0000gp/T/tmpsu3j8yhy/gallery-dl/twitter/UAPJames/UAPJames-1869141126051217764-(20241217_220226)-img1.mp4'],
            # 'error': None
            # }

            for file_path in result["local_files"]:
                try:
                    files.append(discord.File(file_path))
                    logger.debug(f"Added file to upload: {file_path}")
                except Exception as ex:
                    logger.warning(f"Failed to create discord.File for {file_path}: {ex}")
                    print(f"{ex}")
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    print(f"Error Class: {ex.__class__}")
                    output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                    print(output)
                    print(f"exc_type: {exc_type}")
                    print(f"exc_value: {exc_value}")
                    traceback.print_tb(exc_traceback)
                    if aiosettings.dev_mode:
                        bpdb.pm()

            # Send completion message first
            await ctx.send("Download complete!")
            # Then update progress embed
            await progress.edit(embed=embed)
            # Finally send any files
            if files:
                logger.debug(f"Uploading {len(files)} media files")
                await ctx.send(files=files)
                # Clean up temp directory after files are sent
                if result["local_files"]:
                    logger.debug("Cleaning up temp directory after files are sent")
                    try:
                        self._cleanup_temp_dir(result["local_files"][0])
                    except Exception as e:
                        logger.warning(f"Failed to cleanup temp directory: {e}")
                        return False, "Failed to cleanup temp directory"

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
    async def download(self, ctx: commands.Context, url: str, *args: Any, **kwargs: Any) -> None:
        """Download tweet media and metadata.

        Args:
            ctx: Command context
            url: Tweet URL to download
        """
        try:
            logger.info(f"{type(self).__name__} -> ctx = {ctx}, url = {url}")
            logger.debug(f"Download command invoked - URL: {url}")
            await self._handle_download(ctx, url, mode="single")
            logger.debug("Download command completed")
        except Exception as e:
            logger.exception("Error in download command")
            error_embed = create_error_embed(str(e))
            await ctx.send(embed=error_embed)
            raise TwitterError(f"Failed to download tweet: {e}") from e

    @download.error
    async def download_error_handler(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send(
                embed=discord.Embed(description="Sorry, you need `MANAGE SERVER` permissions to change the download!")
            )

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

    @thread.error
    async def thread_error_handler(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send(
                embed=discord.Embed(description="Sorry, you need `MANAGE SERVER` permissions to change the thread!")
            )

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

    @card.error
    async def card_error_handler(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send(
                embed=discord.Embed(description="Sorry, you need `MANAGE SERVER` permissions to change the card!")
            )

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

    @info.error
    async def info_error_handler(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send(
                embed=discord.Embed(description="Sorry, you need `MANAGE SERVER` permissions to change the info!")
            )


async def setup(bot: commands.Bot) -> None:
    """Add Twitter cog to bot.

    Args:
        bot: Discord bot instance
    """
    logger.debug("Setting up Twitter cog")
    await bot.add_cog(Twitter(bot))
    logger.debug("Twitter cog setup complete")
