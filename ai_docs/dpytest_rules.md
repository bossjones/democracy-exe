I'll revise the plan based on the existing code. Let me analyze the current implementation first.

Read file: democracy_exe/chatbot/cogs/twitter.py


I'll update the PLAN.md to focus on what's actually needed.

```markdown:PLAN.md
```markdown
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
```
```


I've updated the plan to focus on three main areas:

1. **Twitter Cog Edge Cases (2 hours)**
   - Adding proper URL validation
   - Handling rate limiting and timeouts
   - Improving temp file cleanup
   - Better error handling

2. **LangGraph/LangChain Integration (4 hours)**
   - Creating the TwitterTool for LangChain
   - Integrating with the existing LangGraph pipeline
   - Handling media downloads through AI
   - Error handling and logging

3. **Basic Test Coverage (2 hours)**
   - Setting up basic test structure
   - Testing critical paths
   - Testing error handling
   - Aiming for 10-20% coverage on new modules

The plan is now more focused on what's actually needed since you already have the core Twitter cog and bot functionality implemented.

Would you like me to start implementing any of these components? I'd suggest starting with either:
1. The edge case handling in the Twitter cog
2. The LangChain TwitterTool implementation

Which would you prefer to tackle first?
