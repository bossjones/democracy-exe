"""
This type stub file was generated by pyright.
"""

import discord.abc
import discord.utils
from typing import Any, Dict, Generator, Generic, List, Optional, Sequence, TYPE_CHECKING, Type, TypeVar, Union, overload
from discord import Interaction, Message, Permissions, User
from discord.context_managers import Typing
from .view import StringView
from ._types import BotT
from typing_extensions import ParamSpec, Self, TypeGuard
from discord.abc import MessageableChannel
from discord.guild import Guild
from discord.member import Member
from discord.user import ClientUser
from discord.voice_client import VoiceProtocol
from discord.embeds import Embed
from discord.file import File
from discord.mentions import AllowedMentions
from discord.sticker import GuildSticker, StickerItem
from discord.message import MessageReference, PartialMessage
from discord.ui import View
from discord.poll import Poll
from .cog import Cog
from .core import Command
from .parameters import Parameter
from types import TracebackType

"""
This type stub file was generated by pyright.
"""
if TYPE_CHECKING:
    BE = TypeVar('BE', bound=BaseException)
__all__ = ('Context', )
MISSING: Any = ...
T = TypeVar('T')
CogT = TypeVar('CogT', bound="Cog")
if TYPE_CHECKING:
    P = ParamSpec('P')
else:
    ...
def is_cog(obj: Any) -> TypeGuard[Cog]:
    ...

class DeferTyping:
    def __init__(self, ctx: Context[BotT], *, ephemeral: bool) -> None:
        ...
    
    def __await__(self) -> Generator[Any, None, None]:
        ...
    
    async def __aenter__(self) -> None:
        ...
    
    async def __aexit__(self, exc_type: Optional[Type[BE]], exc: Optional[BE], traceback: Optional[TracebackType]) -> None:
        ...
    


class Context(discord.abc.Messageable, Generic[BotT]):
    r"""Represents the context in which a command is being invoked under.

    This class contains a lot of meta data to help you understand more about
    the invocation context. This class is not created manually and is instead
    passed around to commands as the first parameter.

    This class implements the :class:`~discord.abc.Messageable` ABC.

    Attributes
    -----------
    message: :class:`.Message`
        The message that triggered the command being executed.

        .. note::

            In the case of an interaction based context, this message is "synthetic"
            and does not actually exist. Therefore, the ID on it is invalid similar
            to ephemeral messages.
    bot: :class:`.Bot`
        The bot that contains the command being executed.
    args: :class:`list`
        The list of transformed arguments that were passed into the command.
        If this is accessed during the :func:`.on_command_error` event
        then this list could be incomplete.
    kwargs: :class:`dict`
        A dictionary of transformed arguments that were passed into the command.
        Similar to :attr:`args`\, if this is accessed in the
        :func:`.on_command_error` event then this dict could be incomplete.
    current_parameter: Optional[:class:`Parameter`]
        The parameter that is currently being inspected and converted.
        This is only of use for within converters.

        .. versionadded:: 2.0
    current_argument: Optional[:class:`str`]
        The argument string of the :attr:`current_parameter` that is currently being converted.
        This is only of use for within converters.

        .. versionadded:: 2.0
    interaction: Optional[:class:`~discord.Interaction`]
        The interaction associated with this context.

        .. versionadded:: 2.0
    prefix: Optional[:class:`str`]
        The prefix that was used to invoke the command. For interaction based contexts,
        this is ``/`` for slash commands and ``\u200b`` for context menu commands.
    command: Optional[:class:`Command`]
        The command that is being invoked currently.
    invoked_with: Optional[:class:`str`]
        The command name that triggered this invocation. Useful for finding out
        which alias called the command.
    invoked_parents: List[:class:`str`]
        The command names of the parents that triggered this invocation. Useful for
        finding out which aliases called the command.

        For example in commands ``?a b c test``, the invoked parents are ``['a', 'b', 'c']``.

        .. versionadded:: 1.7

    invoked_subcommand: Optional[:class:`Command`]
        The subcommand that was invoked.
        If no valid subcommand was invoked then this is equal to ``None``.
    subcommand_passed: Optional[:class:`str`]
        The string that was attempted to call a subcommand. This does not have
        to point to a valid registered subcommand and could just point to a
        nonsense string. If nothing was passed to attempt a call to a
        subcommand then this is set to ``None``.
    command_failed: :class:`bool`
        A boolean that indicates if the command failed to be parsed, checked,
        or invoked.
    """
    def __init__(self, *, message: Message, bot: BotT, view: StringView, args: List[Any] = ..., kwargs: Dict[str, Any] = ..., prefix: Optional[str] = ..., command: Optional[Command[Any, ..., Any]] = ..., invoked_with: Optional[str] = ..., invoked_parents: List[str] = ..., invoked_subcommand: Optional[Command[Any, ..., Any]] = ..., subcommand_passed: Optional[str] = ..., command_failed: bool = ..., current_parameter: Optional[Parameter] = ..., current_argument: Optional[str] = ..., interaction: Optional[Interaction[BotT]] = ...) -> None:
        ...
    
    @classmethod
    async def from_interaction(cls, interaction: Interaction[BotT], /) -> Self:
        """|coro|

        Creates a context from a :class:`discord.Interaction`. This only
        works on application command based interactions, such as slash commands
        or context menus.

        On slash command based interactions this creates a synthetic :class:`~discord.Message`
        that points to an ephemeral message that the command invoker has executed. This means
        that :attr:`Context.author` returns the member that invoked the command.

        In a message context menu based interaction, the :attr:`Context.message` attribute
        is the message that the command is being executed on. This means that :attr:`Context.author`
        returns the author of the message being targetted. To get the member that invoked
        the command then :attr:`discord.Interaction.user` should be used instead.

        .. versionadded:: 2.0

        Parameters
        -----------
        interaction: :class:`discord.Interaction`
            The interaction to create a context with.

        Raises
        -------
        ValueError
            The interaction does not have a valid command.
        TypeError
            The interaction client is not derived from :class:`Bot` or :class:`AutoShardedBot`.
        """
        ...
    
    async def invoke(self, command: Command[CogT, P, T], /, *args: P.args, **kwargs: P.kwargs) -> T:
        r"""|coro|

        Calls a command with the arguments given.

        This is useful if you want to just call the callback that a
        :class:`.Command` holds internally.

        .. note::

            This does not handle converters, checks, cooldowns, pre-invoke,
            or after-invoke hooks in any matter. It calls the internal callback
            directly as-if it was a regular function.

            You must take care in passing the proper arguments when
            using this function.

        .. versionchanged:: 2.0

            ``command`` parameter is now positional-only.

        Parameters
        -----------
        command: :class:`.Command`
            The command that is going to be called.
        \*args
            The arguments to use.
        \*\*kwargs
            The keyword arguments to use.

        Raises
        -------
        TypeError
            The command argument to invoke is missing.
        """
        ...
    
    async def reinvoke(self, *, call_hooks: bool = ..., restart: bool = ...) -> None:
        """|coro|

        Calls the command again.

        This is similar to :meth:`~.Context.invoke` except that it bypasses
        checks, cooldowns, and error handlers.

        .. note::

            If you want to bypass :exc:`.UserInputError` derived exceptions,
            it is recommended to use the regular :meth:`~.Context.invoke`
            as it will work more naturally. After all, this will end up
            using the old arguments the user has used and will thus just
            fail again.

        Parameters
        ------------
        call_hooks: :class:`bool`
            Whether to call the before and after invoke hooks.
        restart: :class:`bool`
            Whether to start the call chain from the very beginning
            or where we left off (i.e. the command that caused the error).
            The default is to start where we left off.

        Raises
        -------
        ValueError
            The context to reinvoke is not valid.
        """
        ...
    
    @property
    def valid(self) -> bool:
        """:class:`bool`: Checks if the invocation context is valid to be invoked with."""
        ...
    
    @property
    def clean_prefix(self) -> str:
        """:class:`str`: The cleaned up invoke prefix. i.e. mentions are ``@name`` instead of ``<@id>``.

        .. versionadded:: 2.0
        """
        ...
    
    @property
    def cog(self) -> Optional[Cog]:
        """Optional[:class:`.Cog`]: Returns the cog associated with this context's command. None if it does not exist."""
        ...
    
    @property
    def filesize_limit(self) -> int:
        """:class:`int`: Returns the maximum number of bytes files can have when uploaded to this guild or DM channel associated with this context.

        .. versionadded:: 2.3
        """
        ...
    
    @discord.utils.cached_property
    def guild(self) -> Optional[Guild]:
        """Optional[:class:`.Guild`]: Returns the guild associated with this context's command. None if not available."""
        ...
    
    @discord.utils.cached_property
    def channel(self) -> MessageableChannel:
        """Union[:class:`.abc.Messageable`]: Returns the channel associated with this context's command.
        Shorthand for :attr:`.Message.channel`.
        """
        ...
    
    @discord.utils.cached_property
    def author(self) -> Union[User, Member]:
        """Union[:class:`~discord.User`, :class:`.Member`]:
        Returns the author associated with this context's command. Shorthand for :attr:`.Message.author`
        """
        ...
    
    @discord.utils.cached_property
    def me(self) -> Union[Member, ClientUser]:
        """Union[:class:`.Member`, :class:`.ClientUser`]:
        Similar to :attr:`.Guild.me` except it may return the :class:`.ClientUser` in private message contexts.
        """
        ...
    
    @discord.utils.cached_property
    def permissions(self) -> Permissions:
        """:class:`.Permissions`: Returns the resolved permissions for the invoking user in this channel.
        Shorthand for :meth:`.abc.GuildChannel.permissions_for` or :attr:`.Interaction.permissions`.

        .. versionadded:: 2.0
        """
        ...
    
    @discord.utils.cached_property
    def bot_permissions(self) -> Permissions:
        """:class:`.Permissions`: Returns the resolved permissions for the bot in this channel.
        Shorthand for :meth:`.abc.GuildChannel.permissions_for` or :attr:`.Interaction.app_permissions`.

        For interaction-based commands, this will reflect the effective permissions
        for :class:`Context` calls, which may differ from calls through
        other :class:`.abc.Messageable` endpoints, like :attr:`channel`.

        Notably, sending messages, embedding links, and attaching files are always
        permitted, while reading messages might not be.

        .. versionadded:: 2.0
        """
        ...
    
    @property
    def voice_client(self) -> Optional[VoiceProtocol]:
        r"""Optional[:class:`.VoiceProtocol`]: A shortcut to :attr:`.Guild.voice_client`\, if applicable."""
        ...
    
    async def send_help(self, *args: Any) -> Any:
        """send_help(entity=<bot>)

        |coro|

        Shows the help command for the specified entity if given.
        The entity can be a command or a cog.

        If no entity is given, then it'll show help for the
        entire bot.

        If the entity is a string, then it looks up whether it's a
        :class:`Cog` or a :class:`Command`.

        .. note::

            Due to the way this function works, instead of returning
            something similar to :meth:`~.commands.HelpCommand.command_not_found`
            this returns ``None`` on bad input or no help command.

        Parameters
        ------------
        entity: Optional[Union[:class:`Command`, :class:`Cog`, :class:`str`]]
            The entity to show help for.

        Returns
        --------
        Any
            The result of the help command, if any.
        """
        ...
    
    @overload
    async def reply(self, content: Optional[str] = ..., *, tts: bool = ..., embed: Embed = ..., file: File = ..., stickers: Sequence[Union[GuildSticker, StickerItem]] = ..., delete_after: float = ..., nonce: Union[str, int] = ..., allowed_mentions: AllowedMentions = ..., reference: Union[Message, MessageReference, PartialMessage] = ..., mention_author: bool = ..., view: View = ..., suppress_embeds: bool = ..., ephemeral: bool = ..., silent: bool = ..., poll: Poll = ...) -> Message:
        ...
    
    @overload
    async def reply(self, content: Optional[str] = ..., *, tts: bool = ..., embed: Embed = ..., files: Sequence[File] = ..., stickers: Sequence[Union[GuildSticker, StickerItem]] = ..., delete_after: float = ..., nonce: Union[str, int] = ..., allowed_mentions: AllowedMentions = ..., reference: Union[Message, MessageReference, PartialMessage] = ..., mention_author: bool = ..., view: View = ..., suppress_embeds: bool = ..., ephemeral: bool = ..., silent: bool = ..., poll: Poll = ...) -> Message:
        ...
    
    @overload
    async def reply(self, content: Optional[str] = ..., *, tts: bool = ..., embeds: Sequence[Embed] = ..., file: File = ..., stickers: Sequence[Union[GuildSticker, StickerItem]] = ..., delete_after: float = ..., nonce: Union[str, int] = ..., allowed_mentions: AllowedMentions = ..., reference: Union[Message, MessageReference, PartialMessage] = ..., mention_author: bool = ..., view: View = ..., suppress_embeds: bool = ..., ephemeral: bool = ..., silent: bool = ..., poll: Poll = ...) -> Message:
        ...
    
    @overload
    async def reply(self, content: Optional[str] = ..., *, tts: bool = ..., embeds: Sequence[Embed] = ..., files: Sequence[File] = ..., stickers: Sequence[Union[GuildSticker, StickerItem]] = ..., delete_after: float = ..., nonce: Union[str, int] = ..., allowed_mentions: AllowedMentions = ..., reference: Union[Message, MessageReference, PartialMessage] = ..., mention_author: bool = ..., view: View = ..., suppress_embeds: bool = ..., ephemeral: bool = ..., silent: bool = ..., poll: Poll = ...) -> Message:
        ...
    
    async def reply(self, content: Optional[str] = ..., **kwargs: Any) -> Message:
        """|coro|

        A shortcut method to :meth:`send` to reply to the
        :class:`~discord.Message` referenced by this context.

        For interaction based contexts, this is the same as :meth:`send`.

        .. versionadded:: 1.6

        .. versionchanged:: 2.0
            This function will now raise :exc:`TypeError` or
            :exc:`ValueError` instead of ``InvalidArgument``.

        Raises
        --------
        ~discord.HTTPException
            Sending the message failed.
        ~discord.Forbidden
            You do not have the proper permissions to send the message.
        ValueError
            The ``files`` list is not of the appropriate size
        TypeError
            You specified both ``file`` and ``files``.

        Returns
        ---------
        :class:`~discord.Message`
            The message that was sent.
        """
        ...
    
    def typing(self, *, ephemeral: bool = ...) -> Union[Typing, DeferTyping]:
        """Returns an asynchronous context manager that allows you to send a typing indicator to
        the destination for an indefinite period of time, or 10 seconds if the context manager
        is called using ``await``.

        In an interaction based context, this is equivalent to a :meth:`defer` call and
        does not do any typing calls.

        Example Usage: ::

            async with channel.typing():
                # simulate something heavy
                await asyncio.sleep(20)

            await channel.send('Done!')

        Example Usage: ::

            await channel.typing()
            # Do some computational magic for about 10 seconds
            await channel.send('Done!')

        .. versionchanged:: 2.0
            This no longer works with the ``with`` syntax, ``async with`` must be used instead.

        .. versionchanged:: 2.0
            Added functionality to ``await`` the context manager to send a typing indicator for 10 seconds.

        Parameters
        -----------
        ephemeral: :class:`bool`
            Indicates whether the deferred message will eventually be ephemeral.
            Only valid for interaction based contexts.

            .. versionadded:: 2.0
        """
        ...
    
    async def defer(self, *, ephemeral: bool = ...) -> None:
        """|coro|

        Defers the interaction based contexts.

        This is typically used when the interaction is acknowledged
        and a secondary action will be done later.

        If this isn't an interaction based context then it does nothing.

        Parameters
        -----------
        ephemeral: :class:`bool`
            Indicates whether the deferred message will eventually be ephemeral.

        Raises
        -------
        HTTPException
            Deferring the interaction failed.
        InteractionResponded
            This interaction has already been responded to before.
        """
        ...
    
    @overload
    async def send(self, content: Optional[str] = ..., *, tts: bool = ..., embed: Embed = ..., file: File = ..., stickers: Sequence[Union[GuildSticker, StickerItem]] = ..., delete_after: float = ..., nonce: Union[str, int] = ..., allowed_mentions: AllowedMentions = ..., reference: Union[Message, MessageReference, PartialMessage] = ..., mention_author: bool = ..., view: View = ..., suppress_embeds: bool = ..., ephemeral: bool = ..., silent: bool = ..., poll: Poll = ...) -> Message:
        ...
    
    @overload
    async def send(self, content: Optional[str] = ..., *, tts: bool = ..., embed: Embed = ..., files: Sequence[File] = ..., stickers: Sequence[Union[GuildSticker, StickerItem]] = ..., delete_after: float = ..., nonce: Union[str, int] = ..., allowed_mentions: AllowedMentions = ..., reference: Union[Message, MessageReference, PartialMessage] = ..., mention_author: bool = ..., view: View = ..., suppress_embeds: bool = ..., ephemeral: bool = ..., silent: bool = ..., poll: Poll = ...) -> Message:
        ...
    
    @overload
    async def send(self, content: Optional[str] = ..., *, tts: bool = ..., embeds: Sequence[Embed] = ..., file: File = ..., stickers: Sequence[Union[GuildSticker, StickerItem]] = ..., delete_after: float = ..., nonce: Union[str, int] = ..., allowed_mentions: AllowedMentions = ..., reference: Union[Message, MessageReference, PartialMessage] = ..., mention_author: bool = ..., view: View = ..., suppress_embeds: bool = ..., ephemeral: bool = ..., silent: bool = ..., poll: Poll = ...) -> Message:
        ...
    
    @overload
    async def send(self, content: Optional[str] = ..., *, tts: bool = ..., embeds: Sequence[Embed] = ..., files: Sequence[File] = ..., stickers: Sequence[Union[GuildSticker, StickerItem]] = ..., delete_after: float = ..., nonce: Union[str, int] = ..., allowed_mentions: AllowedMentions = ..., reference: Union[Message, MessageReference, PartialMessage] = ..., mention_author: bool = ..., view: View = ..., suppress_embeds: bool = ..., ephemeral: bool = ..., silent: bool = ..., poll: Poll = ...) -> Message:
        ...
    
    async def send(self, content: Optional[str] = ..., *, tts: bool = ..., embed: Optional[Embed] = ..., embeds: Optional[Sequence[Embed]] = ..., file: Optional[File] = ..., files: Optional[Sequence[File]] = ..., stickers: Optional[Sequence[Union[GuildSticker, StickerItem]]] = ..., delete_after: Optional[float] = ..., nonce: Optional[Union[str, int]] = ..., allowed_mentions: Optional[AllowedMentions] = ..., reference: Optional[Union[Message, MessageReference, PartialMessage]] = ..., mention_author: Optional[bool] = ..., view: Optional[View] = ..., suppress_embeds: bool = ..., ephemeral: bool = ..., silent: bool = ..., poll: Poll = ...) -> Message:
        """|coro|

        Sends a message to the destination with the content given.

        This works similarly to :meth:`~discord.abc.Messageable.send` for non-interaction contexts.

        For interaction based contexts this does one of the following:

        - :meth:`discord.InteractionResponse.send_message` if no response has been given.
        - A followup message if a response has been given.
        - Regular send if the interaction has expired

        .. versionchanged:: 2.0
            This function will now raise :exc:`TypeError` or
            :exc:`ValueError` instead of ``InvalidArgument``.

        Parameters
        ------------
        content: Optional[:class:`str`]
            The content of the message to send.
        tts: :class:`bool`
            Indicates if the message should be sent using text-to-speech.
        embed: :class:`~discord.Embed`
            The rich embed for the content.
        file: :class:`~discord.File`
            The file to upload.
        files: List[:class:`~discord.File`]
            A list of files to upload. Must be a maximum of 10.
        nonce: :class:`int`
            The nonce to use for sending this message. If the message was successfully sent,
            then the message will have a nonce with this value.
        delete_after: :class:`float`
            If provided, the number of seconds to wait in the background
            before deleting the message we just sent. If the deletion fails,
            then it is silently ignored.
        allowed_mentions: :class:`~discord.AllowedMentions`
            Controls the mentions being processed in this message. If this is
            passed, then the object is merged with :attr:`~discord.Client.allowed_mentions`.
            The merging behaviour only overrides attributes that have been explicitly passed
            to the object, otherwise it uses the attributes set in :attr:`~discord.Client.allowed_mentions`.
            If no object is passed at all then the defaults given by :attr:`~discord.Client.allowed_mentions`
            are used instead.

            .. versionadded:: 1.4

        reference: Union[:class:`~discord.Message`, :class:`~discord.MessageReference`, :class:`~discord.PartialMessage`]
            A reference to the :class:`~discord.Message` to which you are replying, this can be created using
            :meth:`~discord.Message.to_reference` or passed directly as a :class:`~discord.Message`. You can control
            whether this mentions the author of the referenced message using the :attr:`~discord.AllowedMentions.replied_user`
            attribute of ``allowed_mentions`` or by setting ``mention_author``.

            This is ignored for interaction based contexts.

            .. versionadded:: 1.6

        mention_author: Optional[:class:`bool`]
            If set, overrides the :attr:`~discord.AllowedMentions.replied_user` attribute of ``allowed_mentions``.
            This is ignored for interaction based contexts.

            .. versionadded:: 1.6
        view: :class:`discord.ui.View`
            A Discord UI View to add to the message.

            .. versionadded:: 2.0
        embeds: List[:class:`~discord.Embed`]
            A list of embeds to upload. Must be a maximum of 10.

            .. versionadded:: 2.0
        stickers: Sequence[Union[:class:`~discord.GuildSticker`, :class:`~discord.StickerItem`]]
            A list of stickers to upload. Must be a maximum of 3. This is ignored for interaction based contexts.

            .. versionadded:: 2.0
        suppress_embeds: :class:`bool`
            Whether to suppress embeds for the message. This sends the message without any embeds if set to ``True``.

            .. versionadded:: 2.0
        ephemeral: :class:`bool`
            Indicates if the message should only be visible to the user who started the interaction.
            If a view is sent with an ephemeral message and it has no timeout set then the timeout
            is set to 15 minutes. **This is only applicable in contexts with an interaction**.

            .. versionadded:: 2.0
        silent: :class:`bool`
            Whether to suppress push and desktop notifications for the message. This will increment the mention counter
            in the UI, but will not actually send a notification.

            .. versionadded:: 2.2

        poll: :class:`~discord.Poll`
            The poll to send with this message.

            .. versionadded:: 2.4

        Raises
        --------
        ~discord.HTTPException
            Sending the message failed.
        ~discord.Forbidden
            You do not have the proper permissions to send the message.
        ValueError
            The ``files`` list is not of the appropriate size.
        TypeError
            You specified both ``file`` and ``files``,
            or you specified both ``embed`` and ``embeds``,
            or the ``reference`` object is not a :class:`~discord.Message`,
            :class:`~discord.MessageReference` or :class:`~discord.PartialMessage`.

        Returns
        ---------
        :class:`~discord.Message`
            The message that was sent.
        """
        ...
    


