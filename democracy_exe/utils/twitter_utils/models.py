# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Data models for Twitter content."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union

from pydantic import BaseModel, Field, field_validator
from pydantic._internal import _utils

from democracy_exe.utils.twitter_utils.types import TweetDownloadMode


# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false


class TweetDict(TypedDict, total=False):
    """Type hints for tweet dictionary representation."""

    id: str
    author: str
    content: str
    created_at: str
    url: str
    media_urls: list[str]
    retweet_count: int
    like_count: int
    reply_count: int
    card_url: str
    card_description: str
    card_image: str


class ThreadDict(TypedDict):
    """Type hints for thread dictionary representation."""

    author: str
    created_at: str
    tweets: list[TweetDict]


class DownloadDict(TypedDict, total=False):
    """Type hints for download result dictionary representation."""

    success: bool
    mode: str
    local_files: list[str]
    error: str
    metadata: TweetDict | ThreadDict


class MediaType(str, Enum):
    """Types of media that can be attached to tweets."""

    IMAGE = "image"
    VIDEO = "video"
    GIF = "gif"
    AUDIO = "audio"

    @classmethod
    def from_mime_type(cls, mime_type: str) -> MediaType:
        """Get media type from MIME type.

        Args:
            mime_type: MIME type string

        Returns:
            Corresponding media type

        Raises:
            ValueError: If MIME type is not supported
        """
        # Handle exact matches first
        if mime_type == "image/gif":
            return MediaType.GIF

        # Then handle prefix matches
        mime_map = {
            "image/": MediaType.IMAGE,
            "video/": MediaType.VIDEO,
            "audio/": MediaType.AUDIO,
        }
        for mime_prefix, media_type in mime_map.items():
            if mime_type.startswith(mime_prefix):
                return media_type
        raise ValueError(f"Unsupported MIME type: {mime_type}")


@dataclass
class MediaItem:
    """Represents a media item from a tweet.

    Attributes:
        url: Original URL of the media
        type: Type of media (image, video, etc.)
        local_path: Path where media is stored locally
        size: Size in bytes
        width: Width in pixels (if applicable)
        height: Height in pixels (if applicable)
        duration: Duration in seconds (if applicable)
    """

    url: str
    type: MediaType
    local_path: Path | None = None
    size: int | None = None
    width: int | None = None
    height: int | None = None
    duration: float | None = None

    @property
    def is_downloaded(self) -> bool:
        """Check if media has been downloaded.

        Returns:
            True if media has been downloaded
        """
        return self.local_path is not None and self.local_path.exists()

    @property
    def dimensions(self) -> tuple[int, int] | None:
        """Get media dimensions if available.

        Returns:
            Tuple of (width, height) if available
        """
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None


@dataclass
class TweetCard:
    """Represents a Twitter card (e.g. for links).

    Attributes:
        url: URL the card points to
        title: Card title
        description: Card description
        image_url: URL of card image
        local_image: Local path to downloaded card image
    """

    url: str
    title: str
    description: str | None = None
    image_url: str | None = None
    local_image: Path | None = None

    @property
    def has_image(self) -> bool:
        """Check if card has an image.

        Returns:
            True if card has a non-empty image URL
        """
        return bool(self.image_url)

    @property
    def is_downloaded(self) -> bool:
        """Check if card image has been downloaded.

        Returns:
            True if image has been downloaded
        """
        return self.local_image is not None and self.local_image.exists()


@dataclass
class Tweet:
    """Represents a single tweet.

    Attributes:
        id: Tweet ID
        author: Author's username
        content: Tweet text content
        created_at: Creation timestamp
        url: URL to the tweet
        media: List of media items
        card: Twitter card if present
        retweet_count: Number of retweets
        like_count: Number of likes
        reply_count: Number of replies
        quoted_tweet: Quoted tweet if present
    """

    id: str
    author: str
    content: str
    created_at: datetime
    url: str
    media: list[MediaItem] = field(default_factory=list)
    card: TweetCard | None = None
    retweet_count: int = 0
    like_count: int = 0
    reply_count: int = 0
    quoted_tweet: Tweet | None = None

    @property
    def has_media(self) -> bool:
        """Check if tweet has media attachments.

        Returns:
            True if tweet has media
        """
        return bool(self.media and len(self.media) > 0)

    @property
    def has_card(self) -> bool:
        """Check if tweet has a card.

        Returns:
            True if tweet has a card
        """
        return self.card is not None

    @property
    def is_quote(self) -> bool:
        """Check if tweet is a quote tweet.

        Returns:
            True if tweet is a quote
        """
        return self.quoted_tweet is not None

    def to_dict(self) -> TweetDict:
        """Convert tweet to dictionary format.

        Returns:
            Dictionary representation of tweet
        """
        result: TweetDict = {
            "id": self.id,
            "author": self.author,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "url": self.url,
            "retweet_count": self.retweet_count,
            "like_count": self.like_count,
            "reply_count": self.reply_count,
        }

        if self.media:
            result["media_urls"] = [m.url for m in self.media]

        if self.card:
            result["card_url"] = self.card.url
            result["card_description"] = self.card.description
            if self.card.image_url:
                result["card_image"] = self.card.image_url

        return result


@dataclass
class TweetThread:
    """Represents a thread of tweets.

    Attributes:
        tweets: List of tweets in the thread
        author: Thread author
        created_at: Creation time of first tweet
    """

    tweets: list[Tweet]
    author: str
    created_at: datetime

    @property
    def length(self) -> int:
        """Get number of tweets in thread.

        Returns:
            Number of tweets
        """
        return len(self.tweets) if self.tweets else 0

    @property
    def first_tweet(self) -> Tweet | None:
        """Get first tweet in thread.

        Returns:
            First tweet if thread is not empty
        """
        return self.tweets[0] if self.tweets else None

    @property
    def last_tweet(self) -> Tweet | None:
        """Get last tweet in thread.

        Returns:
            Last tweet if thread is not empty
        """
        return self.tweets[-1] if self.tweets else None

    def to_dict(self) -> ThreadDict:
        """Convert thread to dictionary format.

        Returns:
            Dictionary representation of thread
        """
        return ThreadDict(
            author=self.author,
            created_at=self.created_at.isoformat(),
            tweets=[t.to_dict() for t in self.tweets],
        )


@dataclass
class DownloadedContent:
    """Represents downloaded Twitter content.

    Attributes:
        mode: Download mode used
        content: Downloaded tweet or thread
        local_files: List of local file paths
        error: Error message if download failed
    """

    mode: TweetDownloadMode
    content: Tweet | TweetThread | None = None
    local_files: list[Path] = field(default_factory=list)
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if download was successful.

        Returns:
            True if content was downloaded successfully
        """
        return self.content is not None and not self.error

    @property
    def has_files(self) -> bool:
        """Check if content has local files.

        Returns:
            True if there are local files, False if files list is empty or None
        """
        return bool(self.local_files and len(self.local_files) > 0)

    def to_dict(self) -> DownloadDict:
        """Convert download result to dictionary format.

        Returns:
            Dictionary representation of download result
        """
        result: DownloadDict = {
            "success": self.success,
            "mode": self.mode,
            "local_files": [str(f) for f in self.local_files],
        }

        if self.error:
            result["error"] = self.error
        elif self.content:
            result["metadata"] = self.content.to_dict()

        return result




class TwitterUser(BaseModel):
    """Model representing a Twitter user's profile data."""
    id: int | Any = Field(description="The unique identifier for the user")
    name: str = Field(description="The user's screen name/handle")
    nick: str = Field(description="The user's display name")
    location: str | None = Field(None, description="The user's location")
    date: datetime | str | Any = Field(description="The date the user's account was created")
    verified: bool = Field(False, description="Whether the user is verified")
    protected: bool = Field(False, description="Whether the user's tweets are protected")
    profile_banner: str | None = Field(None, description="URL to the user's profile banner image")
    profile_image: str = Field(description="URL to the user's profile image")
    favourites_count: int | Any = Field(description="Number of tweets the user has liked")
    followers_count: int | Any = Field(description="Number of followers the user has")
    friends_count: int | Any = Field(description="Number of users the user is following")
    listed_count: int | Any = Field(description="Number of lists the user is a member of")
    media_count: int | Any = Field(description="Number of media items the user has posted")
    statuses_count: int | Any = Field(description="Total number of tweets by the user")
    description: str = Field(description="The user's profile description/bio")
    url: str | None = Field(None, description="The user's website URL")

    @field_validator('date', mode='before')
    @classmethod
    def parse_date(cls, v: Any) -> Any:
        """Parse date strings that start with 'dt:' prefix.

        Args:
            v: The value to parse

        Returns:
            Parsed datetime or original value
        """
        if isinstance(v, str):
            if v.startswith('dt:'):
                return datetime.fromisoformat(v[3:])
        return v


class TwitterMention(BaseModel):
    """Model representing a user mention in a tweet."""
    id: int | Any = Field(description="The unique identifier of the mentioned user")
    name: str = Field(description="The screen name of the mentioned user")
    nick: str = Field(description="The display name of the mentioned user")


class TweetInfo(BaseModel):
    """Model representing a Twitter tweet info.json file ."""
    tweet_id: int | Any = Field(description="The unique identifier of the tweet")
    retweet_id: int | Any = Field(0, description="The ID of the original tweet if this is a retweet")
    quote_id: int | Any = Field(0, description="The ID of the quoted tweet if this is a quote tweet")
    reply_id: int | Any | None = Field(None, description="The ID of the tweet this is replying to")
    conversation_id: int | Any = Field(description="The ID of the conversation thread")
    date: datetime | str | Any = Field(description="The timestamp when the tweet was created")
    author: TwitterUser = Field(description="The user who authored the tweet")
    user: TwitterUser = Field(description="The user in whose timeline this tweet appears")
    lang: str = Field(description="The language code of the tweet")
    source: str = Field(description="The application used to post the tweet")
    sensitive: bool = Field(False, description="Whether the tweet contains sensitive content")
    favorite_count: int | Any = Field(0, description="Number of likes the tweet has received")
    quote_count: int | Any = Field(0, description="Number of times the tweet has been quoted")
    reply_count: int | Any = Field(0, description="Number of replies to the tweet")
    retweet_count: int | Any = Field(0, description="Number of times the tweet has been retweeted")
    bookmark_count: int | Any = Field(0, description="Number of times the tweet has been bookmarked")
    view_count: int | Any = Field(0, description="Number of times the tweet has been viewed")
    mentions: list[TwitterMention] | None = Field(None, description="List of users mentioned in the tweet")
    content: str = Field(description="The text content of the tweet")
    reply_to: str | None = Field(None, description="The screen name of the user being replied to")
    count: int | Any | None = Field(None, description="Number of media items in the tweet")
    category: str = Field("twitter", description="The category identifier for the tweet")
    subcategory: str = Field(description="The subcategory identifier for the tweet")
    type: str | None = Field(None, description="Type of media in the tweet")
    date_original: datetime | str | None = Field(None, description="Original tweet date for retweets")
    birdwatch: str | None = Field(None, description="Birdwatch note content if present")
    description: str | None = Field(None, description="Media description/alt text")

    @field_validator('date', 'date_original', mode='before')
    @classmethod
    def parse_dates(cls, v: Any) -> Any:
        """Parse date strings that start with 'dt:' prefix.

        Args:
            v: The value to parse

        Returns:
            Parsed datetime or original value
        """
        if isinstance(v, str):
            if v.startswith('dt:'):
                return datetime.fromisoformat(v[3:])
        return v

    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow extra fields that might be present in test data


# from datetime import datetime
# from typing import List, Optional, Union

# from pydantic import BaseModel, Field


# class TweetMetadata(BaseModel):
#     """Model representing metadata for a tweet."""
#     id: Optional[str] = Field(None, description="The unique identifier of the tweet")
#     url: Optional[str] = Field(None, description="The URL of the tweet")
#     author: Optional[str] = Field(None, description="The author of the tweet")
#     content: Optional[str] = Field(None, description="The text content of the tweet")
#     media_urls: Optional[List[str]] = Field(default_factory=list, description="List of media URLs in the tweet")
#     created_at: Optional[str] = Field(None, description="The timestamp when the tweet was created")


# class TweetMetadataReturnModel(BaseModel):
#     """Model representing the return value for tweet metadata operations."""
#     success: bool = Field(True, description="Whether the operation was successful")
#     metadata: Optional[TweetMetadata] = Field(None, description="The tweet metadata if available")
#     local_files: Optional[List[str]] = Field(default_factory=list, description="List of local file paths")
#     error: Optional[str] = Field(None, description="Error message if operation failed")

#     class Config:
#         """Pydantic config."""
#         arbitrary_types_allowed = True
