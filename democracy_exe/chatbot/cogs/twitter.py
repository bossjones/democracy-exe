"""Twitter cog for Discord bot."""
from __future__ import annotations

import asyncio
import pathlib
import tempfile

from typing import Any, Optional

import discord

from discord.ext import commands
from loguru import logger
from rich.pretty import pprint

from democracy_exe.utils.twitter_utils.download import download_tweet
from democracy_exe.utils.twitter_utils.types import DownloadResult, TweetDownloadMode


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
        ctx: commands.Context,
        url: str,
        mode: TweetDownloadMode
    ) -> None:
        """Handle tweet download and send response.

        Args:
            ctx: Command context
            url: Tweet URL
            mode: Download mode (single/thread/card)
        """
        async with ctx.typing():
            # Create progress message
            progress = await ctx.send(f"Downloading tweet {mode}...")

            try:
                # Download tweet content
                result = await download_tweet(url, mode=mode)

                if not result["success"]:
                    await progress.edit(content=f"Failed to download tweet: {result['error']}")
                    return

                # Create embed with tweet info
                embed = discord.Embed(title="Tweet Download")
                embed.add_field(name="Author", value=result["metadata"]["author"])
                embed.add_field(name="Created", value=result["metadata"]["created_at"])
                embed.description = result["metadata"]["content"]

                # Upload media files
                files = []
                for file_path in result["local_files"]:
                    files.append(discord.File(file_path))

                await progress.edit(content="Download complete!", embed=embed)
                if files:
                    await ctx.send(files=files)

            except Exception as e:
                logger.exception("Error downloading tweet")
                await progress.edit(content=f"Error: {e!s}")

    @commands.group(name="tweet")
    async def tweet(self, ctx: commands.Context) -> None:
        """Twitter command group."""
        if ctx.invoked_subcommand is None:
            await ctx.send("Invalid tweet command. Use !help tweet")

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
        """
        async with ctx.typing():
            try:
                result = await download_tweet(url, mode="single")

                if not result["success"]:
                    await ctx.send(f"Failed to get tweet info: {result['error']}")
                    return

                # Create detailed embed
                embed = discord.Embed(title="Tweet Information")
                embed.add_field(name="ID", value=result["metadata"]["id"])
                embed.add_field(name="Author", value=result["metadata"]["author"])
                embed.add_field(name="Created", value=result["metadata"]["created_at"])
                embed.add_field(name="URL", value=result["metadata"]["url"])
                embed.description = result["metadata"]["content"]

                if result["metadata"]["media_urls"]:
                    embed.add_field(
                        name="Media URLs",
                        value="\n".join(result["metadata"]["media_urls"])
                    )

                await ctx.send(embed=embed)

            except Exception as e:
                logger.exception("Error getting tweet info")
                await ctx.send(f"Error: {e!s}")


async def setup(bot: commands.Bot) -> None:
    """Add Twitter cog to bot."""
    await bot.add_cog(Twitter(bot))
