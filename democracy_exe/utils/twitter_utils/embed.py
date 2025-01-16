# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
"""Discord embed utilities for Twitter content."""
from __future__ import annotations

from typing import Any, Dict, Final, List, Optional, TypedDict, Union

import discord
import structlog


logger = structlog.get_logger(__name__)

from democracy_exe.utils.twitter_utils.types import DownloadResult, TweetDownloadMode


# Color constants
BLUE: Final[discord.Color] = discord.Color.blue()
GOLD: Final[discord.Color] = discord.Color.gold()
RED: Final[discord.Color] = discord.Color.red()


class TweetMetadata(TypedDict, total=False):
    """Type hints for tweet metadata."""

    id: str
    author: str
    content: str
    created_at: str
    url: str
    media_urls: list[str]
    retweet_count: int
    like_count: int
    reply_count: int
    card_url: str
    card_description: str
    card_image: str


def _add_optional_field(embed: discord.Embed, metadata: dict[str, Any], field: str, *, display_name: str | None = None, inline: bool = True) -> None:
    """Add field to embed if it exists in metadata.

    Args:
        embed: Discord embed to modify
        metadata: Metadata dictionary
        field: Field name to check and add
        display_name: Optional display name override
        inline: Whether the field should be inline
    """
    if field in metadata:
        name = display_name if display_name is not None else field.replace("_", " ").title()
        embed.add_field(name=name, value=metadata[field], inline=inline)


def _add_media_field(embed: discord.Embed, metadata: dict[str, Any]) -> None:
    """Add media URLs field to embed if present.

    Args:
        embed: Discord embed to modify
        metadata: Metadata dictionary
    """
    if metadata.get("media_urls"):
        media_list = "\n".join(metadata["media_urls"])
        embed.add_field(name="Media URLs", value=media_list, inline=False)


def create_tweet_embed(metadata: TweetMetadata) -> discord.Embed:
    """Create a Discord embed for a single tweet.

    Args:
        metadata: Tweet metadata dictionary containing author, content, etc.

    Returns:
        Discord embed object
    """
    embed = discord.Embed(title="Tweet", color=BLUE)
    embed.set_author(name=metadata["author"])
    embed.add_field(name="Created", value=metadata["created_at"], inline=True)

    _add_optional_field(embed, metadata, "url", display_name="URL")

    if "content" in metadata:
        embed.description = metadata["content"] # type: ignore

    _add_media_field(embed, metadata)

    return embed


def create_thread_embed(metadata_list: list[TweetMetadata]) -> discord.Embed:
    """Create a Discord embed for a tweet thread.

    Args:
        metadata_list: List of tweet metadata dictionaries

    Returns:
        Discord embed object
    """
    if not metadata_list:
        return discord.Embed(title="Empty Thread", description="No tweets found", color=BLUE)

    # Use first tweet as main content
    main_tweet = metadata_list[0]
    embed = discord.Embed(title="Tweet Thread", color=BLUE)
    embed.set_author(name=main_tweet["author"])

    # Add thread content
    thread_content = []
    for i, tweet in enumerate(metadata_list, 1):
        thread_content.append(f"**Tweet {i}:**\n{tweet.get('content', 'No content')}\n")

    embed.description = "\n".join(thread_content) # type: ignore

    # Add metadata
    embed.add_field(name="Thread Length", value=str(len(metadata_list)), inline=True)
    embed.add_field(name="Created", value=main_tweet["created_at"], inline=True)

    _add_optional_field(embed, main_tweet, "url", inline=False)

    return embed


def create_card_embed(metadata: TweetMetadata) -> discord.Embed:
    """Create a Discord embed for a tweet card preview.

    Args:
        metadata: Tweet metadata dictionary

    Returns:
        Discord embed object
    """
    embed = discord.Embed(title="Tweet Card", color=BLUE)
    embed.set_author(name=metadata["author"])

    if "content" in metadata:
        embed.description = metadata["content"] # type: ignore

    _add_optional_field(embed, metadata, "card_url", display_name="Card URL", inline=False)
    _add_optional_field(embed, metadata, "card_description", display_name="Description", inline=False)

    if metadata.get("card_image"):
        embed.set_image(url=metadata["card_image"])
    else:
        embed.set_image(url=None)

    return embed


def create_info_embed(metadata: TweetMetadata) -> discord.Embed:
    """Create a detailed info embed for a tweet.

    Args:
        metadata: Tweet metadata dictionary

    Returns:
        Discord embed object
    """
    embed = discord.Embed(title="Tweet Information", color=BLUE)

    # Add basic metadata
    embed.add_field(name="ID", value=metadata.get("id", "Unknown"), inline=True)
    embed.add_field(name="Author", value=metadata.get("author", "Unknown"), inline=True)
    embed.add_field(name="Created", value=metadata.get("created_at", "Unknown"), inline=True)

    _add_optional_field(embed, metadata, "url", display_name="URL", inline=False)

    if "content" in metadata:
        embed.description = metadata["content"] # type: ignore

    _add_media_field(embed, metadata)

    # Add engagement metrics
    for field in ["retweet_count", "like_count", "reply_count"]:
        _add_optional_field(embed, metadata, field)

    return embed


def create_download_progress_embed(url: str, mode: TweetDownloadMode) -> discord.Embed:
    """Create a progress embed for tweet download.

    Args:
        url: Tweet URL being downloaded
        mode: Download mode (single/thread/card)

    Returns:
        Discord embed object
    """
    embed = discord.Embed(
        title="Download in Progress",
        description=f"Downloading tweet {mode} from {url}...",
        color=GOLD
    )
    embed.set_footer(text="Please wait...")
    return embed


def create_error_embed(error: str) -> discord.Embed:
    """Create an error embed.

    Args:
        error: Error message

    Returns:
        Discord embed object
    """
    embed = discord.Embed(
        title="Error",
        description=str(error),
        color=RED
    )
    return embed
