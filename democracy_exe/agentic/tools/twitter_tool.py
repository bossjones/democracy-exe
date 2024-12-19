"""Twitter tool for LangChain/LangGraph integration."""
from __future__ import annotations

from typing import Any, Optional, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from democracy_exe.utils.twitter_utils.download import TweetDownloadMode, download_tweet
from democracy_exe.utils.twitter_utils.models import DownloadedContent, Tweet, TweetThread


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
        """
        result = download_tweet(url, mode=TweetDownloadMode(mode))
        if isinstance(result, DownloadedContent):
            return result.content
        return result

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
        """
        result = await download_tweet(url, mode=TweetDownloadMode(mode))
        if isinstance(result, DownloadedContent):
            return result.content
        return result
