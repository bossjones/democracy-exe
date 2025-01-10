# democracy_exe/clients/tweetpik.py
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
import structlog


logger = structlog.get_logger(__name__)

from democracy_exe.aio_settings import aiosettings


class TweetPikClient:
    """Client for interacting with TweetPik API.

    Handles screenshot capture of tweets with configurable styling options.
    Uses settings from aiosettings for default configuration values.
    """

    BASE_URL = "https://tweetpik.com/api/v2/images"

    def __init__(self, api_key: str):
        """Initialize TweetPik client with API key and default config.

        Args:
            api_key: TweetPik API authentication key
        """
        if not api_key:
            api_key = aiosettings.tweetpik_authorization.get_secret_value()  # pylint: disable=no-member

        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": api_key
        }

    def _prepare_payload(self, tweet_url: str, **kwargs) -> dict[str, Any]:
        """Prepare request payload with default and override settings.

        Args:
            tweet_url: URL of the tweet to capture
            **kwargs: Optional overrides for default settings

        Returns:
            Dictionary containing the complete request payload
        """
        payload = {
            "url": tweet_url,
            "theme": aiosettings.tweetpik_theme,
            "dimension": aiosettings.tweetpik_dimension,
            "backgroundColor": aiosettings.tweetpik_background_color,
            "textPrimaryColor": aiosettings.tweetpik_text_primary_color,
            "textSecondaryColor": aiosettings.tweetpik_text_secondary_color,
            "linkColor": aiosettings.tweetpik_link_color,
            "verifiedIconColor": aiosettings.tweetpik_verified_icon_color,
            "displayVerified": aiosettings.tweetpik_display_verified,
            "displayMetrics": aiosettings.tweetpik_display_metrics,
            "displayEmbeds": aiosettings.tweetpik_display_embeds,
            "contentScale": aiosettings.tweetpik_content_scale,
            "contentWidth": aiosettings.tweetpik_content_width,
            "twitterToken": aiosettings.tweetpik_api_key.get_secret_value(), # pylint: disable=no-member
        }
        payload.update(kwargs)
        return payload

    def screenshot_tweet(self, tweet_url: str, **kwargs) -> dict[str, Any]:
        """Capture a screenshot of a tweet synchronously.

        Args:
            tweet_url: URL of the tweet to capture
            **kwargs: Optional configuration overrides

        Returns:
            API response containing screenshot data

        Raises:
            httpx.HTTPError: If the API request fails
        """
        payload = self._prepare_payload(tweet_url, **kwargs)
        with httpx.Client() as client:
            response = client.post(self.BASE_URL, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def screenshot_tweet_async(self, tweet_url: str, **kwargs) -> dict[str, Any]:
        """Capture a screenshot of a tweet asynchronously.

        Args:
            tweet_url: URL of the tweet to capture
            **kwargs: Optional configuration overrides

        Returns:
            API response containing screenshot data

        Raises:
            httpx.HTTPError: If the API request fails
        """
        payload = self._prepare_payload(tweet_url, **kwargs)
        async with httpx.AsyncClient() as client:
            response = await client.post(self.BASE_URL, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
