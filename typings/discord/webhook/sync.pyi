"""
This type stub file was generated by pyright.
"""

import threading
from typing import Any, Dict, List, Literal, Optional, Sequence, TYPE_CHECKING, Tuple, Type, TypeVar, Union, overload
from ..message import Attachment, Message
from ..http import Route
from ..channel import ForumTag
from .async_ import BaseWebhook, _WebhookState
from typing_extensions import Self
from types import TracebackType
from ..file import File
from ..embeds import Embed
from ..poll import Poll
from ..mentions import AllowedMentions
from ..abc import Snowflake
from ..state import ConnectionState
from ..types.webhook import Webhook as WebhookPayload
from ..types.message import Message as MessagePayload
from requests import Session

"""
This type stub file was generated by pyright.
"""
__all__ = ('SyncWebhook', 'SyncWebhookMessage')
_log = ...
if TYPE_CHECKING:
    BE = TypeVar('BE', bound=BaseException)
MISSING: Any = ...
class DeferredLock:
    def __init__(self, lock: threading.Lock) -> None:
        ...
    
    def __enter__(self) -> Self:
        ...
    
    def delay_by(self, delta: float) -> None:
        ...
    
    def __exit__(self, exc_type: Optional[Type[BE]], exc: Optional[BE], traceback: Optional[TracebackType]) -> None:
        ...
    


class WebhookAdapter:
    def __init__(self) -> None:
        ...
    
    def request(self, route: Route, session: Session, *, payload: Optional[Dict[str, Any]] = ..., multipart: Optional[List[Dict[str, Any]]] = ..., files: Optional[Sequence[File]] = ..., reason: Optional[str] = ..., auth_token: Optional[str] = ..., params: Optional[Dict[str, Any]] = ...) -> Any:
        ...
    
    def delete_webhook(self, webhook_id: int, *, token: Optional[str] = ..., session: Session, reason: Optional[str] = ...) -> None:
        ...
    
    def delete_webhook_with_token(self, webhook_id: int, token: str, *, session: Session, reason: Optional[str] = ...) -> None:
        ...
    
    def edit_webhook(self, webhook_id: int, token: str, payload: Dict[str, Any], *, session: Session, reason: Optional[str] = ...) -> WebhookPayload:
        ...
    
    def edit_webhook_with_token(self, webhook_id: int, token: str, payload: Dict[str, Any], *, session: Session, reason: Optional[str] = ...) -> WebhookPayload:
        ...
    
    def execute_webhook(self, webhook_id: int, token: str, *, session: Session, payload: Optional[Dict[str, Any]] = ..., multipart: Optional[List[Dict[str, Any]]] = ..., files: Optional[Sequence[File]] = ..., thread_id: Optional[int] = ..., wait: bool = ...) -> MessagePayload:
        ...
    
    def get_webhook_message(self, webhook_id: int, token: str, message_id: int, *, session: Session, thread_id: Optional[int] = ...) -> MessagePayload:
        ...
    
    def edit_webhook_message(self, webhook_id: int, token: str, message_id: int, *, session: Session, payload: Optional[Dict[str, Any]] = ..., multipart: Optional[List[Dict[str, Any]]] = ..., files: Optional[Sequence[File]] = ..., thread_id: Optional[int] = ...) -> MessagePayload:
        ...
    
    def delete_webhook_message(self, webhook_id: int, token: str, message_id: int, *, session: Session, thread_id: Optional[int] = ...) -> None:
        ...
    
    def fetch_webhook(self, webhook_id: int, token: str, *, session: Session) -> WebhookPayload:
        ...
    
    def fetch_webhook_with_token(self, webhook_id: int, token: str, *, session: Session) -> WebhookPayload:
        ...
    


class _WebhookContext(threading.local):
    adapter: Optional[WebhookAdapter] = ...


_context = ...
class SyncWebhookMessage(Message):
    """Represents a message sent from your webhook.

    This allows you to edit or delete a message sent by your
    webhook.

    This inherits from :class:`discord.Message` with changes to
    :meth:`edit` and :meth:`delete` to work.

    .. versionadded:: 2.0
    """
    _state: _WebhookState
    def edit(self, *, content: Optional[str] = ..., embeds: Sequence[Embed] = ..., embed: Optional[Embed] = ..., attachments: Sequence[Union[Attachment, File]] = ..., allowed_mentions: Optional[AllowedMentions] = ...) -> SyncWebhookMessage:
        """Edits the message.

        .. versionchanged:: 2.0
            This function will now raise :exc:`TypeError` or
            :exc:`ValueError` instead of ``InvalidArgument``.

        Parameters
        ------------
        content: Optional[:class:`str`]
            The content to edit the message with or ``None`` to clear it.
        embeds: List[:class:`Embed`]
            A list of embeds to edit the message with.
        embed: Optional[:class:`Embed`]
            The embed to edit the message with. ``None`` suppresses the embeds.
            This should not be mixed with the ``embeds`` parameter.
        attachments: List[Union[:class:`Attachment`, :class:`File`]]
            A list of attachments to keep in the message as well as new files to upload. If ``[]`` is passed
            then all attachments are removed.

            .. note::

                New files will always appear after current attachments.

            .. versionadded:: 2.0
        allowed_mentions: :class:`AllowedMentions`
            Controls the mentions being processed in this message.
            See :meth:`.abc.Messageable.send` for more information.

        Raises
        -------
        HTTPException
            Editing the message failed.
        Forbidden
            Edited a message that is not yours.
        TypeError
            You specified both ``embed`` and ``embeds``
        ValueError
            The length of ``embeds`` was invalid or
            there was no token associated with this webhook.

        Returns
        --------
        :class:`SyncWebhookMessage`
            The newly edited message.
        """
        ...
    
    def add_files(self, *files: File) -> SyncWebhookMessage:
        r"""Adds new files to the end of the message attachments.

        .. versionadded:: 2.0

        Parameters
        -----------
        \*files: :class:`File`
            New files to add to the message.

        Raises
        -------
        HTTPException
            Editing the message failed.
        Forbidden
            Tried to edit a message that isn't yours.

        Returns
        --------
        :class:`SyncWebhookMessage`
            The newly edited message.
        """
        ...
    
    def remove_attachments(self, *attachments: Attachment) -> SyncWebhookMessage:
        r"""Removes attachments from the message.

        .. versionadded:: 2.0

        Parameters
        -----------
        \*attachments: :class:`Attachment`
            Attachments to remove from the message.

        Raises
        -------
        HTTPException
            Editing the message failed.
        Forbidden
            Tried to edit a message that isn't yours.

        Returns
        --------
        :class:`SyncWebhookMessage`
            The newly edited message.
        """
        ...
    
    def delete(self, *, delay: Optional[float] = ...) -> None:
        """Deletes the message.

        Parameters
        -----------
        delay: Optional[:class:`float`]
            If provided, the number of seconds to wait before deleting the message.
            This blocks the thread.

        Raises
        ------
        Forbidden
            You do not have proper permissions to delete the message.
        NotFound
            The message was deleted already.
        HTTPException
            Deleting the message failed.
        """
        ...
    


class SyncWebhook(BaseWebhook):
    """Represents a synchronous Discord webhook.

    For an asynchronous counterpart, see :class:`Webhook`.

    .. container:: operations

        .. describe:: x == y

            Checks if two webhooks are equal.

        .. describe:: x != y

            Checks if two webhooks are not equal.

        .. describe:: hash(x)

            Returns the webhooks's hash.

    .. versionchanged:: 1.4
        Webhooks are now comparable and hashable.

    Attributes
    ------------
    id: :class:`int`
        The webhook's ID
    type: :class:`WebhookType`
        The type of the webhook.

        .. versionadded:: 1.3

    token: Optional[:class:`str`]
        The authentication token of the webhook. If this is ``None``
        then the webhook cannot be used to make requests.
    guild_id: Optional[:class:`int`]
        The guild ID this webhook is for.
    channel_id: Optional[:class:`int`]
        The channel ID this webhook is for.
    user: Optional[:class:`abc.User`]
        The user this webhook was created by. If the webhook was
        received without authentication then this will be ``None``.
    name: Optional[:class:`str`]
        The default name of the webhook.
    source_guild: Optional[:class:`PartialWebhookGuild`]
        The guild of the channel that this webhook is following.
        Only given if :attr:`type` is :attr:`WebhookType.channel_follower`.

        .. versionadded:: 2.0

    source_channel: Optional[:class:`PartialWebhookChannel`]
        The channel that this webhook is following.
        Only given if :attr:`type` is :attr:`WebhookType.channel_follower`.

        .. versionadded:: 2.0
    """
    __slots__: Tuple[str, ...] = ...
    def __init__(self, data: WebhookPayload, session: Session, token: Optional[str] = ..., state: Optional[Union[ConnectionState, _WebhookState]] = ...) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def url(self) -> str:
        """:class:`str` : Returns the webhook's url."""
        ...
    
    @classmethod
    def partial(cls, id: int, token: str, *, session: Session = ..., bot_token: Optional[str] = ...) -> SyncWebhook:
        """Creates a partial :class:`Webhook`.

        Parameters
        -----------
        id: :class:`int`
            The ID of the webhook.
        token: :class:`str`
            The authentication token of the webhook.
        session: :class:`requests.Session`
            The session to use to send requests with. Note
            that the library does not manage the session and
            will not close it. If not given, the ``requests``
            auto session creation functions are used instead.
        bot_token: Optional[:class:`str`]
            The bot authentication token for authenticated requests
            involving the webhook.

        Returns
        --------
        :class:`SyncWebhook`
            A partial :class:`SyncWebhook`.
            A partial :class:`SyncWebhook` is just a :class:`SyncWebhook` object with an ID and a token.
        """
        ...
    
    @classmethod
    def from_url(cls, url: str, *, session: Session = ..., bot_token: Optional[str] = ...) -> SyncWebhook:
        """Creates a partial :class:`Webhook` from a webhook URL.

        Parameters
        ------------
        url: :class:`str`
            The URL of the webhook.
        session: :class:`requests.Session`
            The session to use to send requests with. Note
            that the library does not manage the session and
            will not close it. If not given, the ``requests``
            auto session creation functions are used instead.
        bot_token: Optional[:class:`str`]
            The bot authentication token for authenticated requests
            involving the webhook.

        Raises
        -------
        ValueError
            The URL is invalid.

        Returns
        --------
        :class:`SyncWebhook`
            A partial :class:`SyncWebhook`.
            A partial :class:`SyncWebhook` is just a :class:`SyncWebhook` object with an ID and a token.
        """
        ...
    
    def fetch(self, *, prefer_auth: bool = ...) -> SyncWebhook:
        """Fetches the current webhook.

        This could be used to get a full webhook from a partial webhook.

        .. note::

            When fetching with an unauthenticated webhook, i.e.
            :meth:`is_authenticated` returns ``False``, then the
            returned webhook does not contain any user information.

        Parameters
        -----------
        prefer_auth: :class:`bool`
            Whether to use the bot token over the webhook token
            if available. Defaults to ``True``.

        Raises
        -------
        HTTPException
            Could not fetch the webhook
        NotFound
            Could not find the webhook by this ID
        ValueError
            This webhook does not have a token associated with it.

        Returns
        --------
        :class:`SyncWebhook`
            The fetched webhook.
        """
        ...
    
    def delete(self, *, reason: Optional[str] = ..., prefer_auth: bool = ...) -> None:
        """Deletes this Webhook.

        Parameters
        ------------
        reason: Optional[:class:`str`]
            The reason for deleting this webhook. Shows up on the audit log.

            .. versionadded:: 1.4
        prefer_auth: :class:`bool`
            Whether to use the bot token over the webhook token
            if available. Defaults to ``True``.

        Raises
        -------
        HTTPException
            Deleting the webhook failed.
        NotFound
            This webhook does not exist.
        Forbidden
            You do not have permissions to delete this webhook.
        ValueError
            This webhook does not have a token associated with it.
        """
        ...
    
    def edit(self, *, reason: Optional[str] = ..., name: Optional[str] = ..., avatar: Optional[bytes] = ..., channel: Optional[Snowflake] = ..., prefer_auth: bool = ...) -> SyncWebhook:
        """Edits this Webhook.

        Parameters
        ------------
        name: Optional[:class:`str`]
            The webhook's new default name.
        avatar: Optional[:class:`bytes`]
            A :term:`py:bytes-like object` representing the webhook's new default avatar.
        channel: Optional[:class:`abc.Snowflake`]
            The webhook's new channel. This requires an authenticated webhook.
        reason: Optional[:class:`str`]
            The reason for editing this webhook. Shows up on the audit log.

            .. versionadded:: 1.4
        prefer_auth: :class:`bool`
            Whether to use the bot token over the webhook token
            if available. Defaults to ``True``.

        Raises
        -------
        HTTPException
            Editing the webhook failed.
        NotFound
            This webhook does not exist.
        ValueError
            This webhook does not have a token associated with it
            or it tried editing a channel without authentication.

        Returns
        --------
        :class:`SyncWebhook`
            The newly edited webhook.
        """
        ...
    
    @overload
    def send(self, content: str = ..., *, username: str = ..., avatar_url: Any = ..., tts: bool = ..., file: File = ..., files: Sequence[File] = ..., embed: Embed = ..., embeds: Sequence[Embed] = ..., allowed_mentions: AllowedMentions = ..., thread: Snowflake = ..., thread_name: str = ..., wait: Literal[True], suppress_embeds: bool = ..., silent: bool = ..., applied_tags: List[ForumTag] = ..., poll: Poll = ...) -> SyncWebhookMessage:
        ...
    
    @overload
    def send(self, content: str = ..., *, username: str = ..., avatar_url: Any = ..., tts: bool = ..., file: File = ..., files: Sequence[File] = ..., embed: Embed = ..., embeds: Sequence[Embed] = ..., allowed_mentions: AllowedMentions = ..., thread: Snowflake = ..., thread_name: str = ..., wait: Literal[False] = ..., suppress_embeds: bool = ..., silent: bool = ..., applied_tags: List[ForumTag] = ..., poll: Poll = ...) -> None:
        ...
    
    def send(self, content: str = ..., *, username: str = ..., avatar_url: Any = ..., tts: bool = ..., file: File = ..., files: Sequence[File] = ..., embed: Embed = ..., embeds: Sequence[Embed] = ..., allowed_mentions: AllowedMentions = ..., thread: Snowflake = ..., thread_name: str = ..., wait: bool = ..., suppress_embeds: bool = ..., silent: bool = ..., applied_tags: List[ForumTag] = ..., poll: Poll = ...) -> Optional[SyncWebhookMessage]:
        """Sends a message using the webhook.

        The content must be a type that can convert to a string through ``str(content)``.

        To upload a single file, the ``file`` parameter should be used with a
        single :class:`File` object.

        If the ``embed`` parameter is provided, it must be of type :class:`Embed` and
        it must be a rich embed type. You cannot mix the ``embed`` parameter with the
        ``embeds`` parameter, which must be a :class:`list` of :class:`Embed` objects to send.

        Parameters
        ------------
        content: :class:`str`
            The content of the message to send.
        wait: :class:`bool`
            Whether the server should wait before sending a response. This essentially
            means that the return type of this function changes from ``None`` to
            a :class:`WebhookMessage` if set to ``True``.
        username: :class:`str`
            The username to send with this message. If no username is provided
            then the default username for the webhook is used.
        avatar_url: :class:`str`
            The avatar URL to send with this message. If no avatar URL is provided
            then the default avatar for the webhook is used. If this is not a
            string then it is explicitly cast using ``str``.
        tts: :class:`bool`
            Indicates if the message should be sent using text-to-speech.
        file: :class:`File`
            The file to upload. This cannot be mixed with ``files`` parameter.
        files: List[:class:`File`]
            A list of files to send with the content. This cannot be mixed with the
            ``file`` parameter.
        embed: :class:`Embed`
            The rich embed for the content to send. This cannot be mixed with
            ``embeds`` parameter.
        embeds: List[:class:`Embed`]
            A list of embeds to send with the content. Maximum of 10. This cannot
            be mixed with the ``embed`` parameter.
        allowed_mentions: :class:`AllowedMentions`
            Controls the mentions being processed in this message.

            .. versionadded:: 1.4
        thread: :class:`~discord.abc.Snowflake`
            The thread to send this message to.

            .. versionadded:: 2.0
        thread_name: :class:`str`
            The thread name to create with this webhook if the webhook belongs
            to a :class:`~discord.ForumChannel`. Note that this is mutually
            exclusive with the ``thread`` parameter, as this will create a
            new thread with the given name.

            .. versionadded:: 2.0
        suppress_embeds: :class:`bool`
            Whether to suppress embeds for the message. This sends the message without any embeds if set to ``True``.

            .. versionadded:: 2.0
        silent: :class:`bool`
            Whether to suppress push and desktop notifications for the message. This will increment the mention counter
            in the UI, but will not actually send a notification.

            .. versionadded:: 2.2
        poll: :class:`Poll`
            The poll to send with this message.

            .. warning::

                When sending a Poll via webhook, you cannot manually end it.

            .. versionadded:: 2.4

        Raises
        --------
        HTTPException
            Sending the message failed.
        NotFound
            This webhook was not found.
        Forbidden
            The authorization token for the webhook is incorrect.
        TypeError
            You specified both ``embed`` and ``embeds`` or ``file`` and ``files``
            or ``thread`` and ``thread_name``.
        ValueError
            The length of ``embeds`` was invalid or
            there was no token associated with this webhook.

        Returns
        ---------
        Optional[:class:`SyncWebhookMessage`]
            If ``wait`` is ``True`` then the message that was sent, otherwise ``None``.
        """
        ...
    
    def fetch_message(self, id: int, /, *, thread: Snowflake = ...) -> SyncWebhookMessage:
        """Retrieves a single :class:`~discord.SyncWebhookMessage` owned by this webhook.

        .. versionadded:: 2.0

        Parameters
        ------------
        id: :class:`int`
            The message ID to look for.
        thread: :class:`~discord.abc.Snowflake`
            The thread to look in.

        Raises
        --------
        ~discord.NotFound
            The specified message was not found.
        ~discord.Forbidden
            You do not have the permissions required to get a message.
        ~discord.HTTPException
            Retrieving the message failed.
        ValueError
            There was no token associated with this webhook.

        Returns
        --------
        :class:`~discord.SyncWebhookMessage`
            The message asked for.
        """
        ...
    
    def edit_message(self, message_id: int, *, content: Optional[str] = ..., embeds: Sequence[Embed] = ..., embed: Optional[Embed] = ..., attachments: Sequence[Union[Attachment, File]] = ..., allowed_mentions: Optional[AllowedMentions] = ..., thread: Snowflake = ...) -> SyncWebhookMessage:
        """Edits a message owned by this webhook.

        This is a lower level interface to :meth:`WebhookMessage.edit` in case
        you only have an ID.

        .. versionadded:: 1.6

        Parameters
        ------------
        message_id: :class:`int`
            The message ID to edit.
        content: Optional[:class:`str`]
            The content to edit the message with or ``None`` to clear it.
        embeds: List[:class:`Embed`]
            A list of embeds to edit the message with.
        embed: Optional[:class:`Embed`]
            The embed to edit the message with. ``None`` suppresses the embeds.
            This should not be mixed with the ``embeds`` parameter.
        attachments: List[Union[:class:`Attachment`, :class:`File`]]
            A list of attachments to keep in the message as well as new files to upload. If ``[]`` is passed
            then all attachments are removed.

            .. versionadded:: 2.0
        allowed_mentions: :class:`AllowedMentions`
            Controls the mentions being processed in this message.
            See :meth:`.abc.Messageable.send` for more information.
        thread: :class:`~discord.abc.Snowflake`
            The thread the webhook message belongs to.

            .. versionadded:: 2.0

        Raises
        -------
        HTTPException
            Editing the message failed.
        Forbidden
            Edited a message that is not yours.
        TypeError
            You specified both ``embed`` and ``embeds``
        ValueError
            The length of ``embeds`` was invalid or
            there was no token associated with this webhook.
        """
        ...
    
    def delete_message(self, message_id: int, /, *, thread: Snowflake = ...) -> None:
        """Deletes a message owned by this webhook.

        This is a lower level interface to :meth:`WebhookMessage.delete` in case
        you only have an ID.

        .. versionadded:: 1.6

        Parameters
        ------------
        message_id: :class:`int`
            The message ID to delete.
        thread: :class:`~discord.abc.Snowflake`
            The thread the webhook message belongs to.

            .. versionadded:: 2.0

        Raises
        -------
        HTTPException
            Deleting the message failed.
        Forbidden
            Deleted a message that is not yours.
        ValueError
            This webhook does not have a token associated with it.
        """
        ...
    


