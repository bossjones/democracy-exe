# DemocracyExe POC Development Plan (Revised)

## Overview
This document outlines the remaining tasks to get the DemocracyExe POC fully operational, focusing on edge cases, LangGraph/LangChain integration, and basic test coverage.

## Timeline
Total estimated time: 8 hours

## 1. Twitter Cog Edge Cases (2 hours)
### Error Handling Improvements
- Add URL validation with proper error messages
- Handle rate limiting scenarios
- Add timeout handling for long-running downloads
- Improve temp file cleanup reliability

### Example Edge Cases to Handle
```python
async def _validate_url(self, url: str) -> bool:
    """Validate tweet URL format.

    Args:
        url: URL to validate

    Returns:
        bool: True if valid, False otherwise
    """
    import re
    pattern = r'https?://(?:www\.)?(?:twitter\.com|x\.com)/\w+/status/\d+'
    return bool(re.match(pattern, url))

async def _handle_download(self, ctx: Context, url: str, mode: TweetDownloadMode) -> tuple[bool, str | None]:
    if not await self._validate_url(url):
        await ctx.send("Invalid tweet URL format")
        return False, "Invalid URL"

    # Add timeout handling
    try:
        async with asyncio.timeout(30):  # 30 second timeout
            result = await download_tweet(url, mode=mode)
    except asyncio.TimeoutError:
        await ctx.send("Download timed out")
        return False, "Timeout"
```

## 2. LangGraph/LangChain Integration (4 hours)
### File Structure
```
democracy_exe/
└── agentic/
    └── tools/
        ├── __init__.py
        ├── twitter_tool.py
        └── tests/
            ├── __init__.py
            └── test_twitter_tool.py
```

### Core Implementation
```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from democracy_exe.utils.twitter_utils.download import download_tweet

class TwitterTool(BaseTool):
    name = "twitter_downloader"
    description = "Download and process tweets from Twitter/X URLs"

    async def _arun(self, url: str) -> str:
        """Process tweet URL."""
        result = await download_tweet(url, mode="single")
        if not result["success"]:
            return f"Failed to process tweet: {result['error']}"

        metadata = result["metadata"]
        return f"Tweet from {metadata['author']}: {metadata['content']}"
```

### Integration Points
- Add URL detection in message processing
- Integrate with existing LangGraph pipeline
- Handle media downloads through AI pipeline
- Add proper error handling and logging

## 3. Basic Test Coverage (2 hours)
### Test Structure
```
tests/
├── conftest.py
├── test_twitter_cog/
│   ├── __init__.py
│   ├── test_url_validation.py
│   └── test_error_handling.py
└── test_twitter_tool/
    ├── __init__.py
    └── test_basic_functionality.py
```

### Example Test Implementation
```python
import pytest
from democracy_exe.agentic.tools.twitter_tool import TwitterTool

@pytest.mark.asyncio
async def test_twitter_tool_basic():
    """Test basic TwitterTool functionality."""
    tool = TwitterTool()
    url = "https://twitter.com/example/status/123456789"

    # Test URL validation
    assert await tool._validate_url(url)

    # Test error handling
    bad_url = "https://twitter.com/bad/url"
    result = await tool._arun(bad_url)
    assert "Failed to process" in result
```

### Test Coverage Goals
- 10-20% coverage for new modules
- Focus on critical path testing
- Test error handling and edge cases
- Basic integration tests

## Success Criteria
- [x] Discord bot runs locally
- [x] Twitter cog downloads tweets
- [ ] Edge cases properly handled
- [ ] LangGraph tool processes Twitter URLs
- [ ] Basic test coverage for new modules

## 4. Twitter Tool Development (3 hours)

Create a LangChain/LangGraph tool that replicates the functionality of the Twitter cog, focusing on core Twitter operations without Discord-specific features. The tool should be implemented by subclassing BaseTool and support both synchronous and asynchronous operations.

### Requirements
- Subclass BaseTool instead of using @tool decorator
- Support both sync (_run) and async (_arun) operations
- Match feature parity with existing Twitter cog functionality:
  - Tweet downloading and processing
  - Thread analysis and summarization
  - Media handling (images, videos)
  - URL validation and error handling
  - Rate limit management
  - Proper cleanup of temporary files

### Implementation Structure
```python
from typing import Optional, Type
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from democracy_exe.utils.twitter_utils.models import Tweet, TweetThread, DownloadedContent
from democracy_exe.utils.twitter_utils.download import download_tweet

class TwitterToolInput(BaseModel):
    url: str = Field(description="Twitter/X post URL to process")
    mode: str = Field(description="Download mode (single/thread)", default="single")

class TwitterTool(BaseTool):
    name: str = "twitter_processor"
    description: str = "Process Twitter/X posts and threads, extracting content and media"
    args_schema: Type[BaseModel] = TwitterToolInput

    def _run(
        self,
        url: str,
        mode: str = "single",
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> DownloadedContent:
        """Process Twitter/X content synchronously."""
        pass

    async def _arun(
        self,
        url: str,
        mode: str = "single",
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> DownloadedContent:
        """Process Twitter/X content asynchronously."""
        pass
```

### Integration Points
- Connect with existing twitter_utils modules
- Utilize current URL validation and download logic
- Maintain error handling patterns
- Preserve media processing capabilities
- Keep temporary file management

### Success Criteria
- [ ] Tool successfully processes both single tweets and threads
- [ ] Maintains feature parity with Discord cog
- [ ] Proper error handling and rate limiting
- [ ] Clean async/sync operation support
- [ ] Reliable temporary file cleanup
- [ ] Basic test coverage

## 5. Prompt Engineering (2 hours)
### File Structure
```
democracy_exe/
└── prompts/
    ├── __init__.py
    ├── base.py
    ├── templates/
    │   ├── __init__.py
    │   └── twitter_analysis.py
    └── tests/
        ├── __init__.py
        └── test_twitter_analysis.py
```

### Core Implementation
```python
from langchain.prompts import ChatPromptTemplate
from democracy_exe.utils.twitter_utils.models import Tweet, TweetThread

class TwitterAnalysisPrompt:
    """Prompt template for analyzing Twitter content.

    Handles both single tweets and thread analysis with configurable
    parameters for tone, depth, and output format.
    """

    @staticmethod
    def create_single_tweet_prompt(tweet: Tweet) -> ChatPromptTemplate:
        """Create prompt for single tweet analysis.

        Args:
            tweet: Tweet object to analyze

        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        return ChatPromptTemplate.from_template("""
        Analyze this tweet from {author}:

        Content: {content}
        Created: {created_at}
        Media: {media_count} items

        Provide:
        1. Main topics/themes
        2. Sentiment analysis
        3. Key points
        4. Related context (if any)

        Format as a concise bullet-point list.
        """)

    @staticmethod
    def create_thread_prompt(thread: TweetThread) -> ChatPromptTemplate:
        """Create prompt for thread analysis.

        Args:
            thread: TweetThread object to analyze

        Returns:
            ChatPromptTemplate: Configured prompt template
        """
        return ChatPromptTemplate.from_template("""
        Analyze this thread by {author} containing {tweet_count} tweets:

        {thread_content}

        Provide:
        1. Thread summary (2-3 sentences)
        2. Main arguments/points
        3. Evidence/sources cited
        4. Key takeaways

        Format as a structured markdown summary.
        """)
```

### Test Implementation
```python
import pytest
from democracy_exe.prompts.templates.twitter_analysis import TwitterAnalysisPrompt
from democracy_exe.utils.twitter_utils.models import Tweet, TweetThread

@pytest.mark.asyncio
async def test_single_tweet_prompt():
    """Test single tweet prompt generation."""
    tweet = Tweet(
        id="123",
        author="test_user",
        content="Test tweet content",
        created_at="2024-01-01",
        url="https://twitter.com/test_user/status/123",
        media=[],
        card=None
    )

    prompt = TwitterAnalysisPrompt.create_single_tweet_prompt(tweet)
    result = prompt.format(
        author=tweet.author,
        content=tweet.content,
        created_at=tweet.created_at,
        media_count=len(tweet.media)
    )

    assert "test_user" in result
    assert "Test tweet content" in result
    assert "2024-01-01" in result
```

### Integration Points
- Connect with TwitterTool for automated analysis
- Add prompt templates to LangGraph pipeline
- Implement caching for repeated analyses
- Add proper error handling and logging

### Success Criteria
- [ ] Prompt templates generate valid LangChain prompts
- [ ] Analysis results are properly structured
- [ ] Edge cases handled (empty tweets, long threads)
- [ ] Test coverage for prompt generation
