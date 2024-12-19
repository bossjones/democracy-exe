"""Tests for TwitterTool."""

from __future__ import annotations

import datetime

from typing import TYPE_CHECKING

from langchain_core.tools import ToolException

import pytest

from pytest_mock import MockerFixture

from democracy_exe.agentic.tools.twitter_tool import TwitterTool
from democracy_exe.utils.twitter_utils.models import DownloadedContent, Tweet, TweetThread


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from vcr.request import Request as VCRRequest

    from pytest_mock.plugin import MockerFixture


# # Apply VCR.py configuration to all tests in this module
# pytestmark = pytest.mark.vcr(
#     filter_headers=["authorization"], match_on=["method", "scheme", "host", "port", "path", "query"]
# )


@pytest.fixture
def twitter_tool() -> TwitterTool:
    """Create TwitterTool instance for testing.

    Returns:
        TwitterTool instance
    """
    return TwitterTool()


@pytest.fixture
def mock_tweet() -> Tweet:
    """Create mock Tweet for testing.

    Returns:
        Mock Tweet instance
    """
    return Tweet(
        id="123",
        author="test_user",
        content="Test tweet",
        created_at="2024-01-01T00:00:00Z",
        url="https://twitter.com/test_user/status/123",
        media=[],
        card=None,
    )


@pytest.fixture
def mock_thread() -> TweetThread:
    """Create mock TweetThread for testing.

    Returns:
        Mock TweetThread instance
    """
    return TweetThread(
        tweets=[
            Tweet(
                id="123",
                author="test_user",
                content="Test tweet 1",
                created_at="2024-01-01T00:00:00Z",
                url="https://twitter.com/test_user/status/123",
                media=[],
                card=None,
            ),
            Tweet(
                id="124",
                author="test_user",
                content="Test tweet 2",
                created_at="2024-01-01T00:00:01Z",
                url="https://twitter.com/test_user/status/124",
                media=[],
                card=None,
            ),
        ],
        author="test_user",
        created_at="2024-01-01T00:00:00Z",
    )


def test_validate_mode(twitter_tool: TwitterTool) -> None:
    """Test mode validation."""
    # Valid modes
    twitter_tool._validate_mode("single")
    twitter_tool._validate_mode("thread")

    # Invalid mode
    with pytest.raises(ValueError, match="Invalid mode"):
        twitter_tool._validate_mode("invalid")


# @pytest.mark.asyncio
# @pytest.mark.asyncio
# @pytest.mark.vcr(
#     filter_headers=["authorization", "Set-Cookie"],
#     match_on=["uri", "method", "path", "body"]
# )
# @pytest.mark.vcronly()
# @pytest.mark.default_cassette("test_run_single_tweet.yaml")
# @pytest.mark.vcr(
#     allow_playback_repeats=True,
#     match_on=["method", "scheme", "port", "path", "query", "body", "headers"],
#     ignore_localhost=False,
# )
@pytest.mark.asyncio
@pytest.mark.skip_until(
    deadline=datetime.datetime(2024, 12, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
async def test_run_single_tweet(
    twitter_tool: TwitterTool,
    # mock_tweet: Tweet,
    mocker: MockerFixture,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture,
    vcr: VCRRequest,
) -> None:
    """Test synchronous download of single tweet."""
    mock_download = mocker.patch(
        "democracy_exe.agentic.tools.twitter_tool.adownload_tweet",
        return_value=DownloadedContent(mode="single", content=mock_tweet, local_files=[], error=None),
    )

    result = await twitter_tool.arun({"url": "https://x.com/Eminitybaba_/status/1868256259251863704"})
    # assert isinstance(result, Tweet)
    assert result.id == "1868256259251863704"
    assert result.author == "Eminitybaba_"


# @pytest.mark.asyncio
# async def test_run_thread(twitter_tool: TwitterTool, mock_thread: TweetThread, mocker: MockerFixture) -> None:
#     """Test synchronous download of tweet thread."""
#     mock_download = mocker.patch(
#         "democracy_exe.agentic.tools.twitter_tool.download_tweet",
#         return_value=DownloadedContent(mode="thread", content=mock_thread, local_files=[], error=None),
#     )

#     result = await twitter_tool.arun({"url": "https://twitter.com/test_user/status/123", "mode": "thread"})
#     assert isinstance(result, TweetThread)
#     assert len(result.tweets) == 2
#     assert result.author == "test_user"


# @pytest.mark.asyncio
# async def test_run_with_error(twitter_tool: TwitterTool, mocker: MockerFixture) -> None:
#     """Test handling of download errors."""
#     mock_download = mocker.patch(
#         "democracy_exe.agentic.tools.twitter_tool.download_tweet",
#         return_value=DownloadedContent(mode="single", content=None, local_files=[], error="Download failed"),
#     )

#     with pytest.raises(ValueError, match=r"Download failed$"):
#         await twitter_tool.arun({"url": "https://twitter.com/test_user/status/123"})


# @pytest.mark.asyncio
# async def test_arun_single_tweet(twitter_tool: TwitterTool, mock_tweet: Tweet, mocker: MockerFixture) -> None:
#     """Test asynchronous download of single tweet."""
#     mock_download = mocker.patch("democracy_exe.agentic.tools.twitter_tool.download_tweet")
#     mock_download.return_value = DownloadedContent(mode="single", content=mock_tweet, local_files=[], error=None)

#     result = await twitter_tool.arun({"url": "https://twitter.com/test_user/status/123"})
#     assert isinstance(result, Tweet)
#     assert result.id == "123"
#     assert result.author == "test_user"


# @pytest.mark.asyncio
# async def test_arun_thread(twitter_tool: TwitterTool, mock_thread: TweetThread, mocker: MockerFixture) -> None:
#     """Test asynchronous download of tweet thread."""
#     mock_download = mocker.patch("democracy_exe.agentic.tools.twitter_tool.download_tweet")
#     mock_download.return_value = DownloadedContent(mode="thread", content=mock_thread, local_files=[], error=None)

#     result = await twitter_tool.arun({"url": "https://twitter.com/test_user/status/123", "mode": "thread"})
#     assert isinstance(result, TweetThread)
#     assert len(result.tweets) == 2
#     assert result.author == "test_user"


# @pytest.mark.asyncio
# async def test_arun_with_error(twitter_tool: TwitterTool, mocker: MockerFixture) -> None:
#     """Test handling of asynchronous download errors."""
#     mock_download = mocker.patch("democracy_exe.agentic.tools.twitter_tool.download_tweet")
#     mock_download.return_value = DownloadedContent(mode="single", content=None, local_files=[], error="Download failed")

#     with pytest.raises(ValueError, match=r"Download failed$"):
#         await twitter_tool.arun({"url": "https://twitter.com/test_user/status/123"})
