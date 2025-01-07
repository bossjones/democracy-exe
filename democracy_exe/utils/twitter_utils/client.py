"""Twitter client utilities using gallery-dl."""
from __future__ import annotations

import re
import sys
import traceback

from collections.abc import Iterator
from datetime import datetime
from typing import Any, Dict, Final, List, Optional, TypedDict, cast, final

# from loguru import logger
import structlog

from gallery_dl.extractor import twitter  # type: ignore


logger = structlog.get_logger(__name__)

from .types import TweetMetadata


class GalleryDLTweetItem(TypedDict, total=True):
    """Type definition for gallery-dl tweet item.

    Attributes:
        tweet_id: Unique identifier of the tweet
        tweet_url: Full URL of the tweet
        content: Tweet text content
        date: Tweet creation datetime
        author: Author information dictionary
        media: Optional list of media attachments
    """

    tweet_id: int
    tweet_url: str
    content: str
    date: datetime
    author: dict[str, str]
    media: list[dict[str, str]] | None


class GalleryDLError(Exception):
    """Base exception for gallery-dl related errors."""


class NoExtractorError(GalleryDLError):
    """Raised when no suitable extractor is found."""


class NoTweetDataError(GalleryDLError):
    """Raised when no tweet data is found."""


@final
class TwitterClient:
    """Client for Twitter interactions using gallery-dl.

    This class provides methods to extract and validate tweet metadata using
    gallery-dl as the backend. It handles authentication and configuration
    for gallery-dl's Twitter extractor.

    Attributes:
        auth_token: Twitter authentication token
        config: Gallery-dl configuration dictionary

    Example:
        >>> client = TwitterClient("your_auth_token")
        >>> metadata = client.get_tweet_metadata("https://twitter.com/user/status/123")
        >>> print(metadata["content"])
    """

    # URL patterns for tweet ID extraction
    URL_PATTERNS: Final[list[str]] = [
        r"twitter\.com/\w+/status/(\d+)",
        r"x\.com/\w+/status/(\d+)"
    ]

    # Gallery-dl configuration keys
    CONFIG_KEYS: Final[dict[str, Any]] = {
        "text-tweets": True,
        "include": "metadata",
        "videos": True
    }

    def __init__(self, auth_token: str) -> None:
        """Initialize Twitter client.

        Args:
            auth_token: Twitter authentication token for API access

        Example:
            >>> client = TwitterClient("your_auth_token")
        """
        self.auth_token = auth_token
        self.config = {
            "extractor": {
                "twitter": {
                    "cookies": {
                        "auth_token": auth_token
                    },
                    **self.CONFIG_KEYS
                }
            }
        }

    @staticmethod
    def extract_tweet_id(url: str) -> str | None:
        """Extract tweet ID from URL.

        Supports both twitter.com and x.com URLs in various formats.

        Args:
            url: Tweet URL to parse (twitter.com or x.com)

        Returns:
            Tweet ID if found, None otherwise

        Example:
            >>> TwitterClient.extract_tweet_id("https://twitter.com/user/status/123")
            '123'
            >>> TwitterClient.extract_tweet_id("invalid_url")
            None
        """
        for pattern in TwitterClient.URL_PATTERNS:
            if match := re.search(pattern, url):
                return match.group(1)
        return None

    def get_tweet_metadata(self, url: str) -> TweetMetadata:
        """Get metadata for a tweet.

        Fetches and parses metadata for a single tweet using gallery-dl.

        Args:
            url: Tweet URL to fetch metadata for

        Returns:
            Parsed tweet metadata

        Raises:
            NoExtractorError: If no suitable extractor is found
            NoTweetDataError: If no tweet data is found
            ValueError: If tweet cannot be parsed
            RuntimeError: If gallery-dl encounters an error

        Example:
            >>> metadata = client.get_tweet_metadata("https://twitter.com/user/status/123")
            >>> print(metadata["content"])
        """
        try:
            # Create extractor instance
            extractor = twitter.TwitterExtractor(url)

            # Configure extractor with our settings
            for key, value in self.config.get("extractor", {}).get("twitter", {}).items():
                setattr(extractor, key, value)

            # Get first (and should be only) item for single tweet
            items = list(extractor)
            if not items:
                raise NoTweetDataError(f"No tweet data found for URL: {url}")

            return self._parse_tweet_item(cast(GalleryDLTweetItem, items[0]))

        except Exception as e:
            logger.exception("Error fetching tweet metadata")
            if isinstance(e, (NoExtractorError, NoTweetDataError)):
                raise
            raise ValueError(f"Failed to fetch tweet: {e!s}") from e

    def get_thread_tweets(self, url: str) -> list[TweetMetadata]:
        """Get metadata for all tweets in a thread.

        Fetches and parses metadata for all tweets in a thread, starting from
        any tweet in the thread.

        Args:
            url: URL of any tweet in the thread

        Returns:
            List of tweet metadata for thread, sorted by creation date

        Raises:
            NoExtractorError: If no suitable extractor is found
            NoTweetDataError: If no tweets are found in thread
            ValueError: If thread cannot be parsed
            RuntimeError: If gallery-dl encounters an error

        Example:
            >>> thread = client.get_thread_tweets("https://twitter.com/user/status/123")
            >>> for tweet in thread:
            ...     print(tweet["content"])
        """
        try:
            extractor = twitter.TwitterExtractor(url)
            for key, value in self.config.get("extractor", {}).get("twitter", {}).items():
                setattr(extractor, key, value)

            thread_tweets = []
            for item in extractor:
                thread_tweets.append(
                    self._parse_tweet_item(cast(GalleryDLTweetItem, item))
                )

            if not thread_tweets:
                raise NoTweetDataError(f"No tweets found in thread at URL: {url}")

            # Sort by date
            return sorted(thread_tweets, key=lambda t: t["created_at"])

        except Exception as e:
            logger.exception("Error fetching thread")
            if isinstance(e, (NoExtractorError, NoTweetDataError)):
                raise
            raise ValueError(f"Failed to fetch thread: {e!s}") from e

    def _parse_tweet_item(self, item: GalleryDLTweetItem) -> TweetMetadata:
        """Parse gallery-dl tweet item into metadata.

        Converts a gallery-dl tweet item into our standardized metadata format.

        Args:
            item: Gallery-dl tweet item to parse

        Returns:
            Parsed tweet metadata in standardized format

        Raises:
            ValueError: If required fields are missing

        Example:
            >>> item = {"tweet_id": 123, "content": "Hello", ...}
            >>> metadata = client._parse_tweet_item(item)
            >>> print(metadata["content"])
        """
        try:
            media_urls: list[str] = []
            if item.get("media"):
                for media in item["media"]:
                    if url := media.get("url"):
                        media_urls.append(url)

            return {
                "id": str(item["tweet_id"]),
                "url": item["tweet_url"],
                "author": item["author"]["name"],
                "content": item["content"],
                "media_urls": media_urls,
                "created_at": item["date"].isoformat()
            }
        except KeyError as e:
            raise ValueError(f"Missing required field in tweet data: {e}") from e

    def validate_tweet(self, url: str) -> bool:
        """Check if a tweet URL exists and is accessible.

        Attempts to fetch tweet metadata to verify if the URL is valid and
        accessible with the current authentication.

        Args:
            url: Tweet URL to validate

        Returns:
            True if tweet exists and is accessible

        Example:
            >>> if client.validate_tweet("https://twitter.com/user/status/123"):
            ...     print("Tweet exists!")
        """
        try:
            self.get_tweet_metadata(url)
            return True
        except (NoExtractorError, NoTweetDataError, ValueError):
            return False
