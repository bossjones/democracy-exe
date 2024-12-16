"""Type definitions for Twitter utilities."""

from __future__ import annotations

from typing import Literal, TypedDict


class TweetMetadata(TypedDict):
    """Metadata for a downloaded tweet."""
    id: str
    url: str
    author: str
    content: str
    media_urls: list[str]
    created_at: str

class DownloadResult(TypedDict):
    """Result of a tweet download operation."""
    success: bool
    metadata: TweetMetadata
    local_files: list[str]
    error: str | None

TweetDownloadMode = Literal["single", "thread", "card"]
