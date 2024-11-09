# democracy_exe/ai/agents/social_media_agent.py
from __future__ import annotations

import asyncio
import io
import json
import os

from typing import List, Tuple

import httpx

from PIL import Image

from democracy_exe.ai.base import AgentState, BaseAgent
from democracy_exe.clients.tweetpik import TweetPikClient
from democracy_exe.shell import run_coroutine_subprocess


class SocialMediaAgent(BaseAgent):
    def __init__(self):
        if "TWEETPIK_API_KEY" not in os.environ:
            raise ValueError("TWEETPIK_API_KEY environment variable is not set")
        self.tweetpik_client = TweetPikClient(os.environ["TWEETPIK_API_KEY"])

    async def take_screenshot(self, tweet_url: str) -> bytes:
        result = await self.tweetpik_client.screenshot_tweet_async(tweet_url)
        async with httpx.AsyncClient() as client:
            response = await client.get(result["url"])
        response.raise_for_status()
        return response.content

    def identify_regions(self, image: Image.Image) -> list[tuple[int, int, int, int]]:
        # This is a placeholder. In a real scenario, you'd use an object detection model to identify important regions
        # For demonstration, we'll return a single region covering the whole image
        return [(0, 0, image.width, image.height)]

    def crop_image(self, image: Image.Image, regions: list[tuple[int, int, int, int]],
                   aspect_ratio: tuple[int, int], target_size: tuple[int, int]) -> Image.Image:
        # For simplicity, we'll just crop to the first region and resize
        region = regions[0]
        cropped = image.crop(region)
        return cropped.resize(target_size)

    # async def download_video(self, tweet_url: str) -> str:
    #     cmd = f"gallery-dl {tweet_url}"
    #     result = await run_coroutine_subprocess(cmd, tweet_url)
    #     # Assuming gallery-dl downloads the video to the current directory
    #     # You might need to parse the output to get the exact file name
    #     return result.strip()

    async def download_video(self, tweet_url: str) -> str:
        cmd = f'gallery-dl --no-mtime -v --write-info-json --write-metadata "{tweet_url}"'
        result = await run_coroutine_subprocess(cmd, tweet_url)

        # Parse the output to find the downloaded file
        lines = result.split('\n')
        downloaded_files = [line.split(' ', 1)[-1] for line in lines if line.startswith('[download] Downloading')]

        if not downloaded_files:
            raise ValueError("No files were downloaded")

        # Get the last jsonloaded file (assuming it's the video)
        video_path = downloaded_files[-1]

        # Read the info JSON file to get additional metadata
        info_json_path = os.path.splitext(video_path)[0] + '.info.json'
        if os.path.exists(info_json_path):
            with open(info_json_path) as f:
                info = json.load(f)

            # You can extract additional information from the info JSON if needed
            # For example: tweet_text = info.get('content', '')

        return video_path

    async def process(self, state: AgentState) -> AgentState:
        try:
            tweet_url = state["query"]

            # Check if it's a video tweet (this is a simplified check)
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
