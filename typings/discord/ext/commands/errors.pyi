"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, List, Optional, TYPE_CHECKING, Tuple, Union
from discord.errors import ClientException, DiscordException
from discord.abc import GuildChannel
from discord.threads import Thread
from discord.types.snowflake import Snowflake, SnowflakeList
from discord.app_commands import AppCommandError
from ._types import BotT
from .context import Context
from .converter import Converter
from .cooldowns import BucketType, Cooldown
from .flags import Flag
from .parameters import Parameter

"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
if TYPE_CHECKING:
    ...
__all__ = ('CommandError', 'MissingRequiredArgument', 'MissingRequiredAttachment', 'BadArgument', 'PrivateMessageOnly', 'NoPrivateMessage', 'CheckFailure', 'CheckAnyFailure', 'CommandNotFound', 'DisabledCommand', 'CommandInvokeError', 'TooManyArguments', 'UserInputError', 'CommandOnCooldown', 'MaxConcurrencyReached', 'NotOwner', 'MessageNotFound', 'ObjectNotFound', 'MemberNotFound', 'GuildNotFound', 'UserNotFound', 'ChannelNotFound', 'ThreadNotFound', 'ChannelNotReadable', 'BadColourArgument', 'BadColorArgument', 'RoleNotFound', 'BadInviteArgument', 'EmojiNotFound', 'GuildStickerNotFound', 'ScheduledEventNotFound', 'PartialEmojiConversionFailure', 'BadBoolArgument', 'MissingRole', 'BotMissingRole', 'MissingAnyRole', 'BotMissingAnyRole', 'MissingPermissions', 'BotMissingPermissions', 'NSFWChannelRequired', 'ConversionError', 'BadUnionArgument', 'BadLiteralArgument', 'ArgumentParsingError', 'UnexpectedQuoteError', 'InvalidEndOfQuotedStringError', 'ExpectedClosingQuoteError', 'ExtensionError', 'ExtensionAlreadyLoaded', 'ExtensionNotLoaded', 'NoEntryPointError', 'ExtensionFailed', 'ExtensionNotFound', 'CommandRegistrationError', 'FlagError', 'BadFlagArgument', 'MissingFlagArgument', 'TooManyFlags', 'MissingRequiredFlag', 'HybridCommandError', 'RangeError')
class CommandError(DiscordException):
    r"""The base exception type for all command related errors.

    This inherits from :exc:`discord.DiscordException`.

    This exception and exceptions inherited from it are handled
    in a special way as they are caught and passed into a special event
    from :class:`.Bot`\, :func:`.on_command_error`.
    """
    def __init__(self, message: Optional[str] = ..., *args: Any) -> None:
        ...
    


class ConversionError(CommandError):
    """Exception raised when a Converter class raises non-CommandError.

    This inherits from :exc:`CommandError`.

    Attributes
    ----------
    converter: :class:`discord.ext.commands.Converter`
        The converter that failed.
    original: :exc:`Exception`
        The original exception that was raised. You can also get this via
        the ``__cause__`` attribute.
    """
    def __init__(self, converter: Converter[Any], original: Exception) -> None:
        ...
    


class UserInputError(CommandError):
    """The base exception type for errors that involve errors
    regarding user input.

    This inherits from :exc:`CommandError`.
    """
    ...


class CommandNotFound(CommandError):
    """Exception raised when a command is attempted to be invoked
    but no command under that name is found.

    This is not raised for invalid subcommands, rather just the
    initial main command that is attempted to be invoked.

    This inherits from :exc:`CommandError`.
    """
    ...


class MissingRequiredArgument(UserInputError):
    """Exception raised when parsing a command and a parameter
    that is required is not encountered.

    This inherits from :exc:`UserInputError`

    Attributes
    -----------
    param: :class:`Parameter`
        The argument that is missing.
    """
    def __init__(self, param: Parameter) -> None:
        ...
    


class MissingRequiredAttachment(UserInputError):
    """Exception raised when parsing a command and a parameter
    that requires an attachment is not given.

    This inherits from :exc:`UserInputError`

    .. versionadded:: 2.0

    Attributes
    -----------
    param: :class:`Parameter`
        The argument that is missing an attachment.
    """
    def __init__(self, param: Parameter) -> None:
        ...
    


class TooManyArguments(UserInputError):
    """Exception raised when the command was passed too many arguments and its
    :attr:`.Command.ignore_extra` attribute was not set to ``True``.

    This inherits from :exc:`UserInputError`
    """
    ...


class BadArgument(UserInputError):
    """Exception raised when a parsing or conversion failure is encountered
    on an argument to pass into a command.

    This inherits from :exc:`UserInputError`
    """
    ...


class CheckFailure(CommandError):
    """Exception raised when the predicates in :attr:`.Command.checks` have failed.

    This inherits from :exc:`CommandError`
    """
    ...


class CheckAnyFailure(CheckFailure):
    """Exception raised when all predicates in :func:`check_any` fail.

    This inherits from :exc:`CheckFailure`.

    .. versionadded:: 1.3

    Attributes
    ------------
    errors: List[:class:`CheckFailure`]
        A list of errors that were caught during execution.
    checks: List[Callable[[:class:`Context`], :class:`bool`]]
        A list of check predicates that failed.
    """
    def __init__(self, checks: List[Callable[[Context[BotT]], bool]], errors: List[CheckFailure]) -> None:
        ...
    


class PrivateMessageOnly(CheckFailure):
    """Exception raised when an operation does not work outside of private
    message contexts.

    This inherits from :exc:`CheckFailure`
    """
    def __init__(self, message: Optional[str] = ...) -> None:
        ...
    


class NoPrivateMessage(CheckFailure):
    """Exception raised when an operation does not work in private message
    contexts.

    This inherits from :exc:`CheckFailure`
    """
    def __init__(self, message: Optional[str] = ...) -> None:
        ...
    


class NotOwner(CheckFailure):
    """Exception raised when the message author is not the owner of the bot.

    This inherits from :exc:`CheckFailure`
    """
    ...


class ObjectNotFound(BadArgument):
    """Exception raised when the argument provided did not match the format
    of an ID or a mention.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 2.0

    Attributes
    -----------
    argument: :class:`str`
        The argument supplied by the caller that was not matched
    """
    def __init__(self, argument: str) -> None:
        ...
    


class MemberNotFound(BadArgument):
    """Exception raised when the member provided was not found in the bot's
    cache.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: :class:`str`
        The member supplied by the caller that was not found
    """
    def __init__(self, argument: str) -> None:
        ...
    


class GuildNotFound(BadArgument):
    """Exception raised when the guild provided was not found in the bot's cache.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.7

    Attributes
    -----------
    argument: :class:`str`
        The guild supplied by the called that was not found
    """
    def __init__(self, argument: str) -> None:
        ...
    


class UserNotFound(BadArgument):
    """Exception raised when the user provided was not found in the bot's
    cache.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: :class:`str`
        The user supplied by the caller that was not found
    """
    def __init__(self, argument: str) -> None:
        ...
    


class MessageNotFound(BadArgument):
    """Exception raised when the message provided was not found in the channel.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: :class:`str`
        The message supplied by the caller that was not found
    """
    def __init__(self, argument: str) -> None:
        ...
    


class ChannelNotReadable(BadArgument):
    """Exception raised when the bot does not have permission to read messages
    in the channel.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: Union[:class:`.abc.GuildChannel`, :class:`.Thread`]
        The channel supplied by the caller that was not readable
    """
    def __init__(self, argument: Union[GuildChannel, Thread]) -> None:
        ...
    


class ChannelNotFound(BadArgument):
    """Exception raised when the bot can not find the channel.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: Union[:class:`int`, :class:`str`]
        The channel supplied by the caller that was not found
    """
    def __init__(self, argument: Union[int, str]) -> None:
        ...
    


class ThreadNotFound(BadArgument):
    """Exception raised when the bot can not find the thread.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 2.0

    Attributes
    -----------
    argument: :class:`str`
        The thread supplied by the caller that was not found
    """
    def __init__(self, argument: str) -> None:
        ...
    


class BadColourArgument(BadArgument):
    """Exception raised when the colour is not valid.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: :class:`str`
        The colour supplied by the caller that was not valid
    """
    def __init__(self, argument: str) -> None:
        ...
    


BadColorArgument = BadColourArgument
class RoleNotFound(BadArgument):
    """Exception raised when the bot can not find the role.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: :class:`str`
        The role supplied by the caller that was not found
    """
    def __init__(self, argument: str) -> None:
        ...
    


class BadInviteArgument(BadArgument):
    """Exception raised when the invite is invalid or expired.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: :class:`str`
        The invite supplied by the caller that was not valid
    """
    def __init__(self, argument: str) -> None:
        ...
    


class EmojiNotFound(BadArgument):
    """Exception raised when the bot can not find the emoji.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: :class:`str`
        The emoji supplied by the caller that was not found
    """
    def __init__(self, argument: str) -> None:
        ...
    


class PartialEmojiConversionFailure(BadArgument):
    """Exception raised when the emoji provided does not match the correct
    format.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: :class:`str`
        The emoji supplied by the caller that did not match the regex
    """
    def __init__(self, argument: str) -> None:
        ...
    


class GuildStickerNotFound(BadArgument):
    """Exception raised when the bot can not find the sticker.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 2.0

    Attributes
    -----------
    argument: :class:`str`
        The sticker supplied by the caller that was not found
    """
    def __init__(self, argument: str) -> None:
        ...
    


class ScheduledEventNotFound(BadArgument):
    """Exception raised when the bot can not find the scheduled event.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 2.0

    Attributes
    -----------
    argument: :class:`str`
        The event supplied by the caller that was not found
    """
    def __init__(self, argument: str) -> None:
        ...
    


class BadBoolArgument(BadArgument):
    """Exception raised when a boolean argument was not convertable.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 1.5

    Attributes
    -----------
    argument: :class:`str`
        The boolean argument supplied by the caller that is not in the predefined list
    """
    def __init__(self, argument: str) -> None:
        ...
    


class RangeError(BadArgument):
    """Exception raised when an argument is out of range.

    This inherits from :exc:`BadArgument`

    .. versionadded:: 2.0

    Attributes
    -----------
    minimum: Optional[Union[:class:`int`, :class:`float`]]
        The minimum value expected or ``None`` if there wasn't one
    maximum: Optional[Union[:class:`int`, :class:`float`]]
        The maximum value expected or ``None`` if there wasn't one
    value: Union[:class:`int`, :class:`float`, :class:`str`]
        The value that was out of range.
    """
    def __init__(self, value: Union[int, float, str], minimum: Optional[Union[int, float]], maximum: Optional[Union[int, float]]) -> None:
        ...
    


class DisabledCommand(CommandError):
    """Exception raised when the command being invoked is disabled.

    This inherits from :exc:`CommandError`
    """
    ...


class CommandInvokeError(CommandError):
    """Exception raised when the command being invoked raised an exception.

    This inherits from :exc:`CommandError`

    Attributes
    -----------
    original: :exc:`Exception`
        The original exception that was raised. You can also get this via
        the ``__cause__`` attribute.
    """
    def __init__(self, e: Exception) -> None:
        ...
    


class CommandOnCooldown(CommandError):
    """Exception raised when the command being invoked is on cooldown.

    This inherits from :exc:`CommandError`

    Attributes
    -----------
    cooldown: :class:`~discord.app_commands.Cooldown`
        A class with attributes ``rate`` and ``per`` similar to the
        :func:`.cooldown` decorator.
    type: :class:`BucketType`
        The type associated with the cooldown.
    retry_after: :class:`float`
        The amount of seconds to wait before you can retry again.
    """
    def __init__(self, cooldown: Cooldown, retry_after: float, type: BucketType) -> None:
        ...
    


class MaxConcurrencyReached(CommandError):
    """Exception raised when the command being invoked has reached its maximum concurrency.

    This inherits from :exc:`CommandError`.

    Attributes
    ------------
    number: :class:`int`
        The maximum number of concurrent invokers allowed.
    per: :class:`.BucketType`
        The bucket type passed to the :func:`.max_concurrency` decorator.
    """
    def __init__(self, number: int, per: BucketType) -> None:
        ...
    


class MissingRole(CheckFailure):
    """Exception raised when the command invoker lacks a role to run a command.

    This inherits from :exc:`CheckFailure`

    .. versionadded:: 1.1

    Attributes
    -----------
    missing_role: Union[:class:`str`, :class:`int`]
        The required role that is missing.
        This is the parameter passed to :func:`~.commands.has_role`.
    """
    def __init__(self, missing_role: Snowflake) -> None:
        ...
    


class BotMissingRole(CheckFailure):
    """Exception raised when the bot's member lacks a role to run a command.

    This inherits from :exc:`CheckFailure`

    .. versionadded:: 1.1

    Attributes
    -----------
    missing_role: Union[:class:`str`, :class:`int`]
        The required role that is missing.
        This is the parameter passed to :func:`~.commands.has_role`.
    """
    def __init__(self, missing_role: Snowflake) -> None:
        ...
    


class MissingAnyRole(CheckFailure):
    """Exception raised when the command invoker lacks any of
    the roles specified to run a command.

    This inherits from :exc:`CheckFailure`

    .. versionadded:: 1.1

    Attributes
    -----------
    missing_roles: List[Union[:class:`str`, :class:`int`]]
        The roles that the invoker is missing.
        These are the parameters passed to :func:`~.commands.has_any_role`.
    """
    def __init__(self, missing_roles: SnowflakeList) -> None:
        ...
    


class BotMissingAnyRole(CheckFailure):
    """Exception raised when the bot's member lacks any of
    the roles specified to run a command.

    This inherits from :exc:`CheckFailure`

    .. versionadded:: 1.1

    Attributes
    -----------
    missing_roles: List[Union[:class:`str`, :class:`int`]]
        The roles that the bot's member is missing.
        These are the parameters passed to :func:`~.commands.has_any_role`.

    """
    def __init__(self, missing_roles: SnowflakeList) -> None:
        ...
    


class NSFWChannelRequired(CheckFailure):
    """Exception raised when a channel does not have the required NSFW setting.

    This inherits from :exc:`CheckFailure`.

    .. versionadded:: 1.1

    Attributes
    -----------
    channel: Union[:class:`.abc.GuildChannel`, :class:`.Thread`]
        The channel that does not have NSFW enabled.
    """
    def __init__(self, channel: Union[GuildChannel, Thread]) -> None:
        ...
    


class MissingPermissions(CheckFailure):
    """Exception raised when the command invoker lacks permissions to run a
    command.

    This inherits from :exc:`CheckFailure`

    Attributes
    -----------
    missing_permissions: List[:class:`str`]
        The required permissions that are missing.
    """
    def __init__(self, missing_permissions: List[str], *args: Any) -> None:
        ...
    


class BotMissingPermissions(CheckFailure):
    """Exception raised when the bot's member lacks permissions to run a
    command.

    This inherits from :exc:`CheckFailure`

    Attributes
    -----------
    missing_permissions: List[:class:`str`]
        The required permissions that are missing.
    """
    def __init__(self, missing_permissions: List[str], *args: Any) -> None:
        ...
    


class BadUnionArgument(UserInputError):
    """Exception raised when a :data:`typing.Union` converter fails for all
    its associated types.

    This inherits from :exc:`UserInputError`

    Attributes
    -----------
    param: :class:`inspect.Parameter`
        The parameter that failed being converted.
    converters: Tuple[Type, ``...``]
        A tuple of converters attempted in conversion, in order of failure.
    errors: List[:class:`CommandError`]
        A list of errors that were caught from failing the conversion.
    """
    def __init__(self, param: Parameter, converters: Tuple[type, ...], errors: List[CommandError]) -> None:
        ...
    


class BadLiteralArgument(UserInputError):
    """Exception raised when a :data:`typing.Literal` converter fails for all
    its associated values.

    This inherits from :exc:`UserInputError`

    .. versionadded:: 2.0

    Attributes
    -----------
    param: :class:`inspect.Parameter`
        The parameter that failed being converted.
    literals: Tuple[Any, ``...``]
        A tuple of values compared against in conversion, in order of failure.
    errors: List[:class:`CommandError`]
        A list of errors that were caught from failing the conversion.
    argument: :class:`str`
        The argument's value that failed to be converted. Defaults to an empty string.

        .. versionadded:: 2.3
    """
    def __init__(self, param: Parameter, literals: Tuple[Any, ...], errors: List[CommandError], argument: str = ...) -> None:
        ...
    


class ArgumentParsingError(UserInputError):
    """An exception raised when the parser fails to parse a user's input.

    This inherits from :exc:`UserInputError`.

    There are child classes that implement more granular parsing errors for
    i18n purposes.
    """
    ...


class UnexpectedQuoteError(ArgumentParsingError):
    """An exception raised when the parser encounters a quote mark inside a non-quoted string.

    This inherits from :exc:`ArgumentParsingError`.

    Attributes
    ------------
    quote: :class:`str`
        The quote mark that was found inside the non-quoted string.
    """
    def __init__(self, quote: str) -> None:
        ...
    


class InvalidEndOfQuotedStringError(ArgumentParsingError):
    """An exception raised when a space is expected after the closing quote in a string
    but a different character is found.

    This inherits from :exc:`ArgumentParsingError`.

    Attributes
    -----------
    char: :class:`str`
        The character found instead of the expected string.
    """
    def __init__(self, char: str) -> None:
        ...
    


class ExpectedClosingQuoteError(ArgumentParsingError):
    """An exception raised when a quote character is expected but not found.

    This inherits from :exc:`ArgumentParsingError`.

    Attributes
    -----------
    close_quote: :class:`str`
        The quote character expected.
    """
    def __init__(self, close_quote: str) -> None:
        ...
    


class ExtensionError(DiscordException):
    """Base exception for extension related errors.

    This inherits from :exc:`~discord.DiscordException`.

    Attributes
    ------------
    name: :class:`str`
        The extension that had an error.
    """
    def __init__(self, message: Optional[str] = ..., *args: Any, name: str) -> None:
        ...
    


class ExtensionAlreadyLoaded(ExtensionError):
    """An exception raised when an extension has already been loaded.

    This inherits from :exc:`ExtensionError`
    """
    def __init__(self, name: str) -> None:
        ...
    


class ExtensionNotLoaded(ExtensionError):
    """An exception raised when an extension was not loaded.

    This inherits from :exc:`ExtensionError`
    """
    def __init__(self, name: str) -> None:
        ...
    


class NoEntryPointError(ExtensionError):
    """An exception raised when an extension does not have a ``setup`` entry point function.

    This inherits from :exc:`ExtensionError`
    """
    def __init__(self, name: str) -> None:
        ...
    


class ExtensionFailed(ExtensionError):
    """An exception raised when an extension failed to load during execution of the module or ``setup`` entry point.

    This inherits from :exc:`ExtensionError`

    Attributes
    -----------
    name: :class:`str`
        The extension that had the error.
    original: :exc:`Exception`
        The original exception that was raised. You can also get this via
        the ``__cause__`` attribute.
    """
    def __init__(self, name: str, original: Exception) -> None:
        ...
    


class ExtensionNotFound(ExtensionError):
    """An exception raised when an extension is not found.

    This inherits from :exc:`ExtensionError`

    .. versionchanged:: 1.3
        Made the ``original`` attribute always None.

    Attributes
    -----------
    name: :class:`str`
        The extension that had the error.
    """
    def __init__(self, name: str) -> None:
        ...
    


class CommandRegistrationError(ClientException):
    """An exception raised when the command can't be added
    because the name is already taken by a different command.

    This inherits from :exc:`discord.ClientException`

    .. versionadded:: 1.4

    Attributes
    ----------
    name: :class:`str`
        The command name that had the error.
    alias_conflict: :class:`bool`
        Whether the name that conflicts is an alias of the command we try to add.
    """
    def __init__(self, name: str, *, alias_conflict: bool = ...) -> None:
        ...
    


class FlagError(BadArgument):
    """The base exception type for all flag parsing related errors.

    This inherits from :exc:`BadArgument`.

    .. versionadded:: 2.0
    """
    ...


class TooManyFlags(FlagError):
    """An exception raised when a flag has received too many values.

    This inherits from :exc:`FlagError`.

    .. versionadded:: 2.0

    Attributes
    ------------
    flag: :class:`~discord.ext.commands.Flag`
        The flag that received too many values.
    values: List[:class:`str`]
        The values that were passed.
    """
    def __init__(self, flag: Flag, values: List[str]) -> None:
        ...
    


class BadFlagArgument(FlagError):
    """An exception raised when a flag failed to convert a value.

    This inherits from :exc:`FlagError`

    .. versionadded:: 2.0

    Attributes
    -----------
    flag: :class:`~discord.ext.commands.Flag`
        The flag that failed to convert.
    argument: :class:`str`
        The argument supplied by the caller that was not able to be converted.
    original: :class:`Exception`
        The original exception that was raised. You can also get this via
        the ``__cause__`` attribute.
    """
    def __init__(self, flag: Flag, argument: str, original: Exception) -> None:
        ...
    


class MissingRequiredFlag(FlagError):
    """An exception raised when a required flag was not given.

    This inherits from :exc:`FlagError`

    .. versionadded:: 2.0

    Attributes
    -----------
    flag: :class:`~discord.ext.commands.Flag`
        The required flag that was not found.
    """
    def __init__(self, flag: Flag) -> None:
        ...
    


class MissingFlagArgument(FlagError):
    """An exception raised when a flag did not get a value.

    This inherits from :exc:`FlagError`

    .. versionadded:: 2.0

    Attributes
    -----------
    flag: :class:`~discord.ext.commands.Flag`
        The flag that did not get a value.
    """
    def __init__(self, flag: Flag) -> None:
        ...
    


class HybridCommandError(CommandError):
    """An exception raised when a :class:`~discord.ext.commands.HybridCommand` raises
    an :exc:`~discord.app_commands.AppCommandError` derived exception that could not be
    sufficiently converted to an equivalent :exc:`CommandError` exception.

    .. versionadded:: 2.0

    Attributes
    -----------
    original: :exc:`~discord.app_commands.AppCommandError`
        The original exception that was raised. You can also get this via
        the ``__cause__`` attribute.
    """
    def __init__(self, original: AppCommandError) -> None:
        ...
    


