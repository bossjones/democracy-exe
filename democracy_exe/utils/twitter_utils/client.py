"""Twitter client utilities using gallery-dl."""
from __future__ import annotations

import re

from collections.abc import Iterator
from typing import Any, Dict, Optional

import gallery_dl

from loguru import logger

from .types import TweetMetadata


class TwitterClient:
    """Client for Twitter interactions using gallery-dl.

    Handles tweet metadata extraction and validation.

    Attributes:
        auth_token: Twitter auth token for authentication
        config: Gallery-dl configuration
    """

    def __init__(self, auth_token: str) -> None:
        """Initialize Twitter client.

        Args:
            auth_token: Twitter authentication token
        """
        self.auth_token = auth_token
        self.config = {
            "extractor": {
                "twitter": {
                    "cookies": {
                        "auth_token": auth_token
                    },
                    "text-tweets": True,
                    "include": "metadata",
                    "videos": True
                }
            }
        }

    @staticmethod
    def extract_tweet_id(url: str) -> str | None:
        """Extract tweet ID from URL.

        Args:
            url: Tweet URL to parse

        Returns:
            Tweet ID if found, None otherwise
        """
        patterns = [
            r"twitter\.com/\w+/status/(\d+)",
            r"x\.com/\w+/status/(\d+)"
        ]

        for pattern in patterns:
            if match := re.search(pattern, url):
                return match.group(1)
        return None

    def get_tweet_metadata(self, url: str) -> TweetMetadata:
        """Get metadata for a tweet.

        Args:
            url: Tweet URL

        Returns:
            Tweet metadata

        Raises:
            ValueError: If tweet cannot be fetched
        """
        try:
            extractor = gallery_dl.extractor.from_url(url, self.config)

            # Get first (and should be only) item for single tweet
            for item in extractor:
                return self._parse_tweet_item(item)

            raise ValueError("No tweet data found")

        except Exception as e:
            logger.exception("Error fetching tweet metadata")
            raise ValueError(f"Failed to fetch tweet: {e!s}") from e

    def get_thread_tweets(self, url: str) -> list[TweetMetadata]:
        """Get metadata for all tweets in a thread.

        Args:
            url: URL of any tweet in the thread

        Returns:
            List of tweet metadata for thread

        Raises:
            ValueError: If thread cannot be fetched
        """
        try:
            extractor = gallery_dl.extractor.from_url(url, self.config)
            thread_tweets = []

            for item in extractor:
                thread_tweets.append(self._parse_tweet_item(item))

            if not thread_tweets:
                raise ValueError("No tweets found in thread")

            # Sort by date
            return sorted(thread_tweets, key=lambda t: t["created_at"])

        except Exception as e:
            logger.exception("Error fetching thread")
            raise ValueError(f"Failed to fetch thread: {e!s}") from e

    def _parse_tweet_item(self, item: dict[str, Any]) -> TweetMetadata:
        """Parse gallery-dl tweet item into metadata.

        Args:
            item: Gallery-dl tweet item

        Returns:
            Parsed tweet metadata
        """
        media_urls = []
        if "media" in item:
            for media in item["media"]:
                if "url" in media:
                    media_urls.append(media["url"])

        return {
            "id": str(item["tweet_id"]),
            "url": item["tweet_url"],
            "author": item["author"]["name"],
            "content": item["content"],
            "media_urls": media_urls,
            "created_at": item["date"].isoformat()
        }

    def validate_tweet(self, url: str) -> bool:
        """Check if a tweet URL exists and is accessible.

        Args:
            url: Tweet URL to validate

        Returns:
            True if tweet exists and is accessible
        """
        try:
            self.get_tweet_metadata(url)
            return True
        except Exception:
            return False
