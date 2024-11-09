# democracy_exe/ai/agents/social_media_agent.py
from __future__ import annotations

import asyncio
import io
import json
import os

from typing import Any, Dict, List, Optional, Tuple

import httpx

from PIL import Image

from democracy_exe.ai.base import AgentState, BaseAgent
from democracy_exe.clients.tweetpik import TweetPikClient
from democracy_exe.shell import run_coroutine_subprocess


class SocialMediaAgent(BaseAgent):
    """Agent for processing social media content, particularly tweets.

    This agent handles screenshot capture, video downloads, and image processing
    for social media content. It uses TweetPik for capturing tweet screenshots
    and gallery-dl for video downloads.

    Raises:
        ValueError: If TWEETPIK_API_KEY environment variable is not set.
    """

    def __init__(self) -> None:
        """Initialize the social media agent with TweetPik client.

        Raises:
            ValueError: If TWEETPIK_API_KEY is not set in environment variables.
        """
        if "TWEETPIK_API_KEY" not in os.environ:
            raise ValueError("TWEETPIK_API_KEY environment variable is not set")
        self.tweetpik_client = TweetPikClient(os.environ["TWEETPIK_API_KEY"])

    def fetch_tweet(self, tweet_url: str) -> dict[str, Any]:
        """Fetch tweet data from URL.

        Args:
            tweet_url: URL of the tweet to fetch

        Returns:
            Dictionary containing tweet data
        """
        return {"url": tweet_url}  # Simplified for now

    def is_video_tweet(self, tweet_data: dict[str, Any]) -> bool:
        """Check if tweet contains video.

        Args:
            tweet_data: Tweet data dictionary

        Returns:
            True if tweet contains video, False otherwise
        """
        return "video" in tweet_data.get("url", "").lower()

    async def take_screenshot(self, tweet_url: str) -> bytes:
        """Capture a screenshot of the tweet.

        Args:
            tweet_url: URL of the tweet to screenshot

        Returns:
            Screenshot image data as bytes

        Raises:
            httpx.HTTPError: If screenshot request fails
        """
        result = await self.tweetpik_client.screenshot_tweet_async(tweet_url)
        async with httpx.AsyncClient() as client:
            response = await client.get(result["url"])
        response.raise_for_status()
        return response.content

    def identify_regions(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        """Identify important regions in the image.

        Args:
            image: PIL Image object to analyze

        Returns:
            List of tuples containing region coordinates (x1, y1, x2, y2)
        """
        # Placeholder implementation
        return [(0, 0, image.width, image.height)]

    def crop_image(
        self,
        image: Image.Image,
        regions: list[tuple[int, int, int, int]],
        aspect_ratio: tuple[int, int],
        target_size: tuple[int, int]
    ) -> Image.Image:
        """Crop image to specified regions and resize.

        Args:
            image: PIL Image object to crop
            regions: List of region coordinates to crop
            aspect_ratio: Desired aspect ratio as (width, height)
            target_size: Target size for the output image

        Returns:
            Cropped and resized PIL Image
        """
        region = regions[0]
        cropped = image.crop(region)
        return cropped.resize(target_size)

    async def download_video(self, tweet_url: str) -> str:
        """Download video from tweet URL using gallery-dl.

        Args:
            tweet_url: URL of the tweet containing video

        Returns:
            Path to downloaded video file

        Raises:
            ValueError: If no files were downloaded
        """
        cmd = f'gallery-dl --no-mtime -v --write-info-json --write-metadata "{tweet_url}"'
        result = await run_coroutine_subprocess(cmd, tweet_url)

        lines = result.split('\n')
        downloaded_files = [line.split(' ', 1)[-1] for line in lines
                          if line.startswith('[download] Downloading')]

        if not downloaded_files:
            raise ValueError("No files were downloaded")

        video_path = downloaded_files[-1]

        # Read the info JSON file to get additional metadata
        info_json_path = os.path.splitext(video_path)[0] + '.info.json'
        if os.path.exists(info_json_path):
            with open(info_json_path) as f:
                info = json.load(f)

        return video_path

    async def process(self, state: AgentState) -> AgentState:
        """Process social media content based on the query.

        Handles both video tweets and regular tweets, downloading videos or
        capturing and processing screenshots as appropriate.

        Args:
            state: Current agent state containing the query (tweet URL)

        Returns:
            Updated agent state with processing results or error message
        """
        try:
            tweet_url = state["query"]

            if "video" in tweet_url.lower():
                video_path = await self.download_video(tweet_url)
                state["response"] = f"Video downloaded and saved to: {video_path}"
            else:
                screenshot_bytes = await self.take_screenshot(tweet_url)
                image = Image.open(io.BytesIO(screenshot_bytes))
                regions = self.identify_regions(image)
                cropped_image = self.crop_image(image, regions, (1, 1), (1080, 1350))
                output_path = "processed_tweet_image.jpg"
                cropped_image.save(output_path)
                state["response"] = f"Tweet image processed and saved to {output_path}"

        except Exception as e:
            state["response"] = f"An error occurred while processing the tweet: {e!s}"

        return state


social_media_agent = SocialMediaAgent()
