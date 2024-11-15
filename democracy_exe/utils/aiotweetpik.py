"""cerebro_bot.utils.aiotweetpik"""
# pylint: disable=unused-import
# NOTE: couple sources
# https://github.com/powerfist01/hawk-eyed/blob/f340c6ff814dd3e2a3cac7a30d03b7c07d95d1e4/services/tweet_to_image/tweetpik.py
# https://github.com/bwhli/birdcatcher/blob/a4b33feff4f2d88d5412cd50b11760312bdd4f1d/app/util/Tweet.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import re
import sys
import typing
import weakref

from collections.abc import Coroutine, Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union
from urllib.parse import quote as _uriquote

import aiofiles
import aiohttp

from loguru import logger

import democracy_exe

from democracy_exe.aio_settings import aiosettings


_from_json = json.loads

TWEETPIK_AUTHORIZATION = aiosettings.tweetpik_authorization.get_secret_value()  # pylint: disable=no-member
TWEETPIK_BUCKET_ID = aiosettings.tweetpik_bucket_id

TWEETPIK_DIMENSION_IG_FEED = aiosettings.tweetpik_dimension_ig_feed
TWEETPIK_DIMENSION_IG_STORY = aiosettings.tweetpik_dimension_ig_story
TWEETPIK_TIMEZONE = aiosettings.tweetpik_timezone
TWEETPIK_DISPLAY_LIKES = aiosettings.tweetpik_display_metrics
TWEETPIK_DISPLAY_REPLIES = aiosettings.tweetpik_display_metrics
TWEETPIK_DISPLAY_RETWEETS = aiosettings.tweetpik_display_metrics
TWEETPIK_DISPLAY_VERIFIED = aiosettings.tweetpik_display_verified
TWEETPIK_DISPLAY_SOURCE = aiosettings.tweetpik_display_embeds
TWEETPIK_DISPLAY_TIME = False
TWEETPIK_DISPLAY_MEDIA_IMAGES = aiosettings.tweetpik_display_media_images
TWEETPIK_DISPLAY_LINK_PREVIEW = aiosettings.tweetpik_display_link_preview
# Any number higher than zero. This value is representing a percentage
TWEETPIK_TEXT_WIDTH = "100"
# Any number higher than zero. This value is used in pixels(px) units
TWEETPIK_CANVAS_WIDTH = "510"

TWEETPIK_BACKGROUND_COLOR = "#FFFFFF"  # Change the background color of the tweet screenshot
TWEETPIK_TEXT_PRIMARY_COLOR = (
    "#000000"  # Change the text primary color used for the main text of the tweet and user's name
)
TWEETPIK_TEXT_SECONDARY_COLOR = (
    "#5B7083"  # Change the text secondary used for the secondary info of the tweet like the username
)
TWEETPIK_LINK_COLOR = "#1B95E0"  # Change the link colors used for the links, hashtags and mentions
TWEETPIK_VERIFIED_ICON = "#1B95E0"  # Change the verified icon color


def _to_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)


if TYPE_CHECKING:
    from aiohttp import ClientResponse

    try:
        from requests import Response

        _ResponseType = Union[ClientResponse, Response]
    except ModuleNotFoundError:
        _ResponseType = ClientResponse

    Snowflake = Union[str, int]
    SnowflakeList = list[Snowflake]

    from types import TracebackType

    T = TypeVar("T")
    BE = TypeVar("BE", bound=BaseException)
    MU = TypeVar("MU", bound="MaybeUnlock")
    Response = Coroutine[Any, Any, T]


class _MissingSentinel:
    def __eq__(self, other):
        return False

    def __bool__(self):
        return False

    def __repr__(self):
        return "..."


MISSING: Any = _MissingSentinel()


def get_tweet_id(tweet_url: str) -> str:
    return re.findall(r"[http?s//]?twitter\.com\/.*\/status\/(\d+)", tweet_url)[0]


def build_tweetpik_download_url(tweetId: str) -> str:
    """Building the URL
    The URL is predictable, so you don't have to worry about storing it. You just need to make sure you generated it before using it. The URL will always consist of your bucket ID and the tweet ID. https://ik.imagekit.io/tweetpik/323251495115948625/tweetId

    Returns:
        str: Url of the image we plan to download
    """
    return f"https://ik.imagekit.io/tweetpik/{TWEETPIK_BUCKET_ID}/{tweetId}"


class TweetpikAPIError(Exception):
    """
    Exception for errors thrown by the API. Contains the HTTP status code and the returned error message.
    """

    def __init__(self, status: int, message: str | dict):
        self.status = status
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        if not isinstance(self.message, str):
            return f"{self.status} {self.message}"
        try:
            self.message = json.loads(self.message)
            return f'{self.status} {self.message["error_summary"]}'
        except Exception:
            return f"{self.status} {self.message}"


class TweetpikException(Exception):
    """Base exception class for tweetpik

    Ideally speaking, this could be caught to handle any exceptions raised from this library.
    """


class ClientException(TweetpikException):
    """Exception that's raised when an operation in the :class:`Client` fails.

    These are usually for exceptions that happened due to user input.
    """


class NoMoreItems(TweetpikException):
    """Exception that is raised when an async iteration operation has no more items."""


class GatewayNotFound(TweetpikException):
    """An exception that is raised when the gateway for Tweetpik could not be found"""

    def __init__(self):
        message = "The gateway to connect to discord was not found."
        super().__init__(message)


def _flatten_error_dict(d: dict[str, Any], key: str = "") -> dict[str, str]:
    items: list[tuple[str, str]] = []
    for k, v in d.items():
        new_key = f"{key}.{k}" if key else k

        if isinstance(v, dict):
            try:
                _errors: list[dict[str, Any]] = v["_errors"]
            except KeyError:
                items.extend(_flatten_error_dict(v, new_key).items())
            else:
                items.append((new_key, " ".join(x.get("message", "") for x in _errors)))
        else:
            items.append((new_key, v))

    return dict(items)


class HTTPException(TweetpikException):
    """Exception that's raised when an HTTP request operation fails.

    Attributes
    ------------
    response: :class:`aiohttp.ClientResponse`
        The response of the failed HTTP request. This is an
        instance of :class:`aiohttp.ClientResponse`. In some cases
        this could also be a :class:`requests.Response`.

    text: :class:`str`
        The text of the error. Could be an empty string.
    status: :class:`int`
        The status code of the HTTP request.
    code: :class:`int`
        The Tweetpik specific error code for the failure.
    """

    def __init__(self, response: _ResponseType, message: str | dict[str, Any] | None):
        self.response: _ResponseType = response
        self.status: int = response.status  # type: ignore
        self.code: int
        self.text: str
        if isinstance(message, dict):
            self.code = message.get("code", 0)
            base = message.get("message", "")
            if errors := message.get("errors"):
                errors = _flatten_error_dict(errors)
                helpful = "\n".join("In %s: %s" % t for t in errors.items())
                self.text = base + "\n" + helpful
            else:
                self.text = base
        else:
            self.text = message or ""
            self.code = 0

        fmt = "{0.status} {0.reason} (error code: {1})"
        if len(self.text):
            fmt += ": {2}"

        super().__init__(fmt.format(self.response, self.code, self.text))


class Forbidden(HTTPException):
    """Exception that's raised for when status code 403 occurs.

    Subclass of :exc:`HTTPException`
    """


class NotFound(HTTPException):
    """Exception that's raised for when status code 404 occurs.

    Subclass of :exc:`HTTPException`
    """


class TweetpikServerError(HTTPException):
    """Exception that's raised for when a 500 range status code occurs.

    Subclass of :exc:`HTTPException`.

    .. versionadded:: 1.5
    """


class InvalidData(ClientException):
    """Exception that's raised when the library encounters unknown
    or invalid data from Tweetpik.
    """


class InvalidArgument(ClientException):
    """Exception that's raised when an argument to a function
    is invalid some way (e.g. wrong value or wrong type).

    This could be considered the analogous of ``ValueError`` and
    ``TypeError`` except inherited from :exc:`ClientException` and thus
    :exc:`TweetpikException`.
    """


class ConnectionClosed(ClientException):
    """Exception that's raised when the gateway connection is
    closed for reasons that could not be handled internally.

    Attributes
    -----------
    code: :class:`int`
        The close code of the websocket.
    reason: :class:`str`
        The reason provided for the closure.
    shard_id: Optional[:class:`int`]
        The shard ID that got closed if applicable.
    """

    def __init__(self, socket: ClientResponse, *, code: int | None = None):
        # This exception is just the same exception except
        # reconfigured to subclass ClientException for users
        self.code: int = code or socket.close_code or -1
        # aiohttp doesn't seem to consistently provide close reason
        self.reason: str = ""
        super().__init__(f"HTTP Request closed with {self.code}")


# SOURCE: discord.py
async def json_or_text(response: aiohttp.ClientResponse) -> dict[str, Any] | str:
    text = await response.text(encoding="utf-8")
    try:
        if response.headers["content-type"] == "application/json":
            return _from_json(text)
    except KeyError:
        # Thanks Cloudflare
        pass

    return text


# SOURCE: discord.py
class TweetpikRoute:
    BASE: ClassVar[str] = "https://tweetpik.com/api"

    def __init__(self, method: str, path: str, **parameters: Any) -> None:
        self.path: str = path
        self.method: str = method
        url = self.BASE + self.path
        if parameters:
            url = url.format_map({k: _uriquote(v) if isinstance(v, str) else v for k, v in parameters.items()})
        self.url: str = url
        self.bucket_id: str = TWEETPIK_BUCKET_ID

    @property
    def bucket(self) -> str:
        # the bucket is just method + path w/ major parameters
        return f"{self.bucket_id}"


# SOURCE: discord.py
class MaybeUnlock:
    def __init__(self, lock: asyncio.Lock) -> None:
        self.lock: asyncio.Lock = lock
        self._unlock: bool = True

    def __enter__(self: MU) -> MU:
        return self

    def defer(self) -> None:
        self._unlock = False

    def __exit__(
        self,
        exc_type: type[BE] | None,
        exc: BE | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._unlock:
            self.lock.release()


# SOURCE: discord.py
class TweetpikHTTPClient:
    """Represents an HTTP client sending HTTP requests to the Tweetpik API."""

    def __init__(
        self,
        connector: aiohttp.BaseConnector | None = None,
        *,
        proxy: str | None = None,
        proxy_auth: aiohttp.BasicAuth | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        unsync_clock: bool = True,
    ) -> None:
        self.loop: asyncio.AbstractEventLoop = asyncio.get_event_loop() if loop is None else loop
        self.connector = connector
        self.__session: aiohttp.ClientSession = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit_per_host=50))
        self._locks: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._global_over: asyncio.Event = asyncio.Event()
        self._global_over.set()
        self.token: str | None = None
        self.bot_token: bool = False
        self.proxy: str | None = proxy
        self.proxy_auth: aiohttp.BasicAuth | None = proxy_auth
        self.use_clock: bool = not unsync_clock

        user_agent = "TweetpikHTTPClient (democracy-exe {0}) Python/{1[0]}.{1[1]} aiohttp/{2}"
        self.user_agent: str = user_agent.format(democracy_exe.__version__, sys.version_info, aiohttp.__version__)

    async def request(
        self,
        route: TweetpikRoute,
        *,
        # files: Optional[Sequence[File]] = None,
        form: Iterable[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        print(f"{form}")

        bucket = route.bucket
        method = route.method
        url = route.url

        lock = self._locks.get(bucket)
        if lock is None:
            lock = asyncio.Lock()
            if bucket is not None:
                self._locks[bucket] = lock

        # header creation
        headers: dict[str, str] = {
            "User-Agent": self.user_agent,
            "Authorization": TWEETPIK_AUTHORIZATION,
            "Content-Type": "application/json",
        }

        kwargs["data"] = _to_json(kwargs.pop("json"))

        kwargs["headers"] = headers

        # Proxy support
        if self.proxy is not None:
            kwargs["proxy"] = self.proxy
        if self.proxy_auth is not None:
            kwargs["proxy_auth"] = self.proxy_auth

        if not self._global_over.is_set():
            # wait until the global lock is complete
            await self._global_over.wait()

        response: aiohttp.ClientResponse | None = None
        data: dict[str, Any] | str | None = None
        await lock.acquire()
        with MaybeUnlock(lock) as maybe_lock:
            for tries in range(5):
                try:
                    async with self.__session.request(method, url, **kwargs) as response:
                        logger.debug(f"{method} {url} with {kwargs.get('data')} has returned {response.status}")

                        # even errors have text involved in them so this is safe to call
                        data = await json_or_text(response)

                        logger.debug("HERE IS THE DATA WE GET BACK FROM THE API CALL BELOVED")
                        logger.debug(data)

                        # the request was successful so just return the text/json
                        if 300 > response.status >= 200:
                            logger.debug(f"{method} {url} has received {data}")
                            return data

                        # we are being rate limited
                        if response.status == 429:
                            if not response.headers.get("Via") or isinstance(data, str):
                                # Banned by Cloudflare more than likely.
                                raise HTTPException(response, data)

                            fmt = 'We are being rate limited. Retrying in %.2f seconds. Handled under the bucket "%s"'

                            # sleep a bit
                            retry_after: float = data["retry_after"]
                            logger.warning(fmt, retry_after, bucket)

                            continue

                        # we've received a 500, 502, or 504, unconditional retry
                        if response.status in {500, 502, 504}:
                            await asyncio.sleep(1 + tries * 2)
                            continue

                        # the usual error cases
                        if response.status == 403:
                            raise Forbidden(response, data)
                        elif response.status == 404:
                            raise NotFound(response, data)
                        elif response.status >= 500:
                            raise TweetpikServerError(response, data)
                        else:
                            raise HTTPException(response, data)

                # This is handling exceptions from the request
                except OSError as e:
                    # Connection reset by peer
                    if tries < 4 and e.errno in (54, 10054):
                        await asyncio.sleep(1 + tries * 2)
                        continue
                    raise

            if response is not None:
                # We've run out of retries, raise.
                if response.status >= 500:
                    raise TweetpikServerError(response, data)

                raise HTTPException(response, data)

            raise RuntimeError("Unreachable code in HTTP handling")

    async def get_from_cdn(self, url: str) -> bytes:
        async with self.__session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            elif resp.status == 404:
                raise NotFound(resp, "asset not found")
            elif resp.status == 403:
                raise Forbidden(resp, "cannot retrieve asset")
            else:
                raise HTTPException(resp, "failed to get asset")

    async def close(self) -> None:
        if self.__session:
            await self.__session.close()

    def images(
        self,
        tweet_url: str | None,
        *,
        dimension_ig_feed: str | None = TWEETPIK_DIMENSION_IG_FEED,
        dimension_ig_story: str | None = TWEETPIK_DIMENSION_IG_STORY,
        timezone: str | None = TWEETPIK_TIMEZONE,
        display_likes: str | None = TWEETPIK_DISPLAY_LIKES,
        display_replies: str | None = TWEETPIK_DISPLAY_REPLIES,
        display_retweets: str | None = TWEETPIK_DISPLAY_RETWEETS,
        display_verified: str | None = TWEETPIK_DISPLAY_VERIFIED,
        display_source: str | None = TWEETPIK_DISPLAY_SOURCE,
        display_time: str | None = TWEETPIK_DISPLAY_TIME,
        display_media_images: str | None = TWEETPIK_DISPLAY_MEDIA_IMAGES,
        display_link_preview: str | None = TWEETPIK_DISPLAY_LINK_PREVIEW,
        text_width: str | None = TWEETPIK_TEXT_WIDTH,
        canvas_width: str | None = TWEETPIK_CANVAS_WIDTH,
        background_color: str | None = TWEETPIK_BACKGROUND_COLOR,
        text_primary_color: str | None = TWEETPIK_TEXT_PRIMARY_COLOR,
        text_secondary_color: str | None = TWEETPIK_TEXT_SECONDARY_COLOR,
        link_color: str | None = TWEETPIK_LINK_COLOR,
        verified_icon: str | None = TWEETPIK_VERIFIED_ICON,
    ) -> Any:
        r = TweetpikRoute("POST", "/images", tweet_url=tweet_url)
        payload = {}

        if tweet_url:
            payload["tweetId"] = get_tweet_id(tweet_url)

        if dimension_ig_feed:
            payload["dimension_ig_feed"] = dimension_ig_feed
        if dimension_ig_story:
            payload["dimension_ig_story"] = dimension_ig_story
        if timezone:
            payload["timezone"] = timezone
        if display_likes:
            payload["display_likes"] = display_likes
        if display_replies:
            payload["display_replies"] = display_replies
        if display_retweets:
            payload["display_retweets"] = display_retweets
        if display_verified:
            payload["display_verified"] = display_verified
        if display_source:
            payload["display_source"] = display_source
        if display_time:
            payload["display_time"] = display_time
        if display_media_images:
            payload["display_media_images"] = display_media_images
        if display_link_preview:
            payload["display_link_preview"] = display_link_preview
        if text_width:
            payload["text_width"] = text_width
        if canvas_width:
            payload["canvas_width"] = canvas_width
        if background_color:
            payload["background_color"] = background_color
        if text_primary_color:
            payload["text_primary_color"] = text_primary_color
        if text_secondary_color:
            payload["text_secondary_color"] = text_secondary_color
        if link_color:
            payload["link_color"] = link_color
        if verified_icon:
            payload["verified_icon"] = verified_icon

        logger.debug("payload debuggggggggggggggggggggggggggg")
        logger.debug(payload)

        return self.request(r, json=payload)

    async def aimages(
        self,
        tweet_url: str | None,
        *,
        dimension_ig_feed: str | None = TWEETPIK_DIMENSION_IG_FEED,
        dimension_ig_story: str | None = TWEETPIK_DIMENSION_IG_STORY,
        timezone: str | None = TWEETPIK_TIMEZONE,
        display_likes: str | None = TWEETPIK_DISPLAY_LIKES,
        display_replies: str | None = TWEETPIK_DISPLAY_REPLIES,
        display_retweets: str | None = TWEETPIK_DISPLAY_RETWEETS,
        display_verified: str | None = TWEETPIK_DISPLAY_VERIFIED,
        display_source: str | None = TWEETPIK_DISPLAY_SOURCE,
        display_time: str | None = TWEETPIK_DISPLAY_TIME,
        display_media_images: str | None = TWEETPIK_DISPLAY_MEDIA_IMAGES,
        display_link_preview: str | None = TWEETPIK_DISPLAY_LINK_PREVIEW,
        text_width: str | None = TWEETPIK_TEXT_WIDTH,
        canvas_width: str | None = TWEETPIK_CANVAS_WIDTH,
        background_color: str | None = TWEETPIK_BACKGROUND_COLOR,
        text_primary_color: str | None = TWEETPIK_TEXT_PRIMARY_COLOR,
        text_secondary_color: str | None = TWEETPIK_TEXT_SECONDARY_COLOR,
        link_color: str | None = TWEETPIK_LINK_COLOR,
        verified_icon: str | None = TWEETPIK_VERIFIED_ICON,
    ) -> Any:
        r = TweetpikRoute("POST", "/images", tweet_url=tweet_url)
        payload = {}

        if tweet_url:
            payload["tweetId"] = get_tweet_id(tweet_url)

        if dimension_ig_feed:
            payload["dimension_ig_feed"] = dimension_ig_feed
        if dimension_ig_story:
            payload["dimension_ig_story"] = dimension_ig_story
        if timezone:
            payload["timezone"] = timezone
        if display_likes:
            payload["display_likes"] = display_likes
        if display_replies:
            payload["display_replies"] = display_replies
        if display_retweets:
            payload["display_retweets"] = display_retweets
        if display_verified:
            payload["display_verified"] = display_verified
        if display_source:
            payload["display_source"] = display_source
        if display_time:
            payload["display_time"] = display_time
        if display_media_images:
            payload["display_media_images"] = display_media_images
        if display_link_preview:
            payload["display_link_preview"] = display_link_preview
        if text_width:
            payload["text_width"] = text_width
        if canvas_width:
            payload["canvas_width"] = canvas_width
        if background_color:
            payload["background_color"] = background_color
        if text_primary_color:
            payload["text_primary_color"] = text_primary_color
        if text_secondary_color:
            payload["text_secondary_color"] = text_secondary_color
        if link_color:
            payload["link_color"] = link_color
        if verified_icon:
            payload["verified_icon"] = verified_icon

        logger.debug("payload debuggggggggggggggggggggggggggg")
        logger.debug(payload)
        data = await self.request(r, json=payload)
        await self.close()

        return data


# TODO: implement multi download https://stackoverflow.com/questions/64282309/aiohttp-download-large-list-of-pdf-files


async def async_download_file(data: dict, dl_dir="./"):
    async with aiohttp.ClientSession() as session:
        url: str = data["url"]
        username: str = data["tweet"]["username"]
        p = pathlib.Path(url)
        p_dl_dir = pathlib.Path(dl_dir)
        full_path_dl_dir = f"{p_dl_dir.absolute()}"
        logger.debug(f"Downloading {url} to {full_path_dl_dir}/{p.name}")
        async with session.get(url) as resp:
            content = await resp.read()

            # Check everything went well
            if resp.status != 200:
                logger.error(f"Download failed: {resp.status}")
                return

            async with aiofiles.open(f"{full_path_dl_dir}/{p.name}", mode="+wb") as f:
                await f.write(content)
