# democracy_exe/clients/tweetpik.py
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class TweetPikClient:
    BASE_URL = "https://tweetpik.com/api/v2/images"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": api_key
        }

    def _prepare_payload(self, tweet_url: str, **kwargs) -> dict[str, Any]:
        payload = {"url": tweet_url}
        payload.update(kwargs)
        return payload

    def screenshot_tweet(self, tweet_url: str, **kwargs) -> dict[str, Any]:
        payload = self._prepare_payload(tweet_url, **kwargs)
        with httpx.Client() as client:
            response = client.post(self.BASE_URL, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    async def screenshot_tweet_async(self, tweet_url: str, **kwargs) -> dict[str, Any]:
        payload = self._prepare_payload(tweet_url, **kwargs)
        async with httpx.AsyncClient() as client:
            response = await client.post(self.BASE_URL, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
