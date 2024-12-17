"""Discord embed utilities for Twitter content."""
from __future__ import annotations


"""
Pseudo-code plan:

- Functions for creating standardized Discord embeds from tweet data
- Support for different embed types (single tweet, thread, card)
- Rich formatting for tweet content
- Media preview handling

Key functions:
    - def create_tweet_embed(metadata: TweetMetadata) -> discord.Embed
    - def create_thread_embed(metadata: list[TweetMetadata]) -> discord.Embed
    - def create_card_embed(metadata: TweetMetadata) -> discord.Embed
"""
