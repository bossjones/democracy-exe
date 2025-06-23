# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
"""Twitter tool for LangChain/LangGraph integration."""
from __future__ import annotations

from typing import Any, Optional, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from democracy_exe.utils.twitter_utils.download import TweetDownloadMode, adownload_tweet, download_tweet
from democracy_exe.utils.twitter_utils.models import DownloadedContent, Tweet, TweetThread
from democracy_exe.utils.twitter_utils.types import DownloadResult


class TwitterToolInput(BaseModel):
    """Schema for Twitter tool input."""
    url: str = Field(description="Twitter/X post URL to process")
    mode: str = Field(
        description="Download mode: 'single' for one tweet, 'thread' for full thread",
        default="single"
    )


class TwitterTool(BaseTool):
    """Tool for downloading and processing Twitter/X content.

    Handles downloading of tweets, threads, and associated media.
    Supports both synchronous and asynchronous operations.

    Attributes:
        name: Tool name for LangChain
        description: Tool description for LLM context
        args_schema: Pydantic model for argument validation
        return_direct: Whether to return results directly to user
    """
    name: str = "twitter_processor"
    description: str = """Process Twitter/X posts. Can download single tweets or full threads.
    Handles text content, media attachments, and Twitter cards. Returns structured content."""
    args_schema: type[BaseModel] = TwitterToolInput
    return_direct: bool = True

    def _validate_mode(self, mode: str) -> None:
        """Validate the download mode.

        Args:
            mode: Download mode to validate

        Raises:
            ValueError: If mode is invalid
        """
        valid_modes = {"single", "thread"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")

    def _run(
        self,
        url: str,
        mode: str = "single",
        run_manager: CallbackManagerForToolRun | None = None
    ) -> Tweet | TweetThread:
        """Download and process Twitter content synchronously.

        Args:
            url: Twitter/X post URL
            mode: Download mode ('single' or 'thread')
            run_manager: Callback manager for tool execution

        Returns:
            Tweet or TweetThread object containing processed content

        Raises:
            ToolException: If download fails or content is invalid
            ValueError: If mode is invalid
        """
        self._validate_mode(mode)
        try:
            result = download_tweet(url, mode=mode)
            if isinstance(result, DownloadedContent):
                if result.error:
                    raise ValueError(f"Download failed: {result.error}")
                return result.content
            return result
        except Exception as e:
            raise ValueError(f"Failed to process Twitter content: {e!s}") from e

    async def _arun(
        self,
        url: str,
        mode: str = "single",
        run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> Tweet | TweetThread:
        """Download and process Twitter content asynchronously.

        Args:
            url: Twitter/X post URL
            mode: Download mode ('single' or 'thread')
            run_manager: Callback manager for async tool execution

        Returns:
            Tweet or TweetThread object containing processed content

        Raises:
            ToolException: If download fails or content is invalid
            ValueError: If mode is invalid
        """
        self._validate_mode(mode)
        try:
            result: DownloadResult = await adownload_tweet(url, mode=mode)
            if isinstance(result, DownloadedContent):
                if result.error:
                    raise ValueError(f"Download failed: {result.error}")
                return result.content
            return result
        except Exception as e:
            raise ValueError(f"Failed to process Twitter content: {e!s}") from e


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Run the Twitter tool asynchronously."""
        tool = TwitterTool()
        result = await tool.arun({"url": "https://x.com/Eminitybaba_/status/1868256259251863704"})
        print(f"Result: {result}")

    asyncio.run(main())
