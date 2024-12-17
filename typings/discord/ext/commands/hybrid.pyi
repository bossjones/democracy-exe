"""
This type stub file was generated by pyright.
"""

import discord
import inspect
from typing import Any, Callable, ClassVar, Dict, List, Optional, TYPE_CHECKING, TypeVar, Union
from discord import app_commands
from .core import Command, Group
from .parameters import Parameter
from .cog import Cog
from typing_extensions import Concatenate, ParamSpec, Self
from ._types import BotT, ContextT, Coro
from .context import Context
from discord.app_commands.commands import AutocompleteCallback, ChoiceT

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
__all__ = ('HybridCommand', 'HybridGroup', 'hybrid_command', 'hybrid_group')
T = TypeVar('T')
U = TypeVar('U')
CogT = TypeVar('CogT', bound='Cog')
CommandT = TypeVar('CommandT', bound='Command[Any, ..., Any]')
GroupT = TypeVar('GroupT', bound='Group[Any, ..., Any]')
_NoneType = ...
if TYPE_CHECKING:
    P = ParamSpec('P')
    P2 = ParamSpec('P2')
    CommandCallback = Union[Callable[Concatenate[CogT, ContextT, P], Coro[T]], Callable[Concatenate[ContextT, P], Coro[T]],]
else:
    ...
class _CallableDefault:
    __slots__ = ...
    def __init__(self, func: Callable[[Context], Any]) -> None:
        ...
    
    @property
    def __class__(self) -> Any:
        ...
    


def is_converter(converter: Any) -> bool:
    ...

def is_transformer(converter: Any) -> bool:
    ...

def required_pos_arguments(func: Callable[..., Any]) -> int:
    ...

class ConverterTransformer(app_commands.Transformer):
    def __init__(self, converter: Any, parameter: Parameter) -> None:
        ...
    
    async def transform(self, interaction: discord.Interaction, value: str, /) -> Any:
        ...
    


class CallableTransformer(app_commands.Transformer):
    def __init__(self, func: Callable[[str], Any]) -> None:
        ...
    
    async def transform(self, interaction: discord.Interaction, value: str, /) -> Any:
        ...
    


class GreedyTransformer(app_commands.Transformer):
    def __init__(self, converter: Any, parameter: Parameter) -> None:
        ...
    
    async def transform(self, interaction: discord.Interaction, value: str, /) -> Any:
        ...
    


def replace_parameter(param: inspect.Parameter, converter: Any, callback: Callable[..., Any], original: Parameter, mapping: Dict[str, inspect.Parameter]) -> inspect.Parameter:
    ...

def replace_parameters(parameters: Dict[str, Parameter], callback: Callable[..., Any], signature: inspect.Signature) -> List[inspect.Parameter]:
    ...

class HybridAppCommand(discord.app_commands.Command[CogT, P, T]):
    __commands_is_hybrid_app_command__: ClassVar[bool] = ...
    def __init__(self, wrapped: Union[HybridCommand[CogT, ..., T], HybridGroup[CogT, ..., T]], name: Optional[Union[str, app_commands.locale_str]] = ...) -> None:
        ...
    
    def copy(self) -> Self:
        ...
    


class HybridCommand(Command[CogT, P, T]):
    r"""A class that is both an application command and a regular text command.

    This has the same parameters and attributes as a regular :class:`~discord.ext.commands.Command`.
    However, it also doubles as an :class:`application command <discord.app_commands.Command>`. In order
    for this to work, the callbacks must have the same subset that is supported by application
    commands.

    These are not created manually, instead they are created via the
    decorator or functional interface.

    .. versionadded:: 2.0
    """
    __commands_is_hybrid__: ClassVar[bool] = ...
    def __init__(self, func: CommandCallback[CogT, Context[Any], P, T], /, *, name: Union[str, app_commands.locale_str] = ..., description: Union[str, app_commands.locale_str] = ..., **kwargs: Any) -> None:
        ...
    
    @property
    def cog(self) -> CogT:
        ...
    
    @cog.setter
    def cog(self, value: CogT) -> None:
        ...
    
    async def can_run(self, ctx: Context[BotT], /) -> bool:
        ...
    
    def autocomplete(self, name: str) -> Callable[[AutocompleteCallback[CogT, ChoiceT]], AutocompleteCallback[CogT, ChoiceT]]:
        """A decorator that registers a coroutine as an autocomplete prompt for a parameter.

        This is the same as :meth:`~discord.app_commands.Command.autocomplete`. It is only
        applicable for the application command and doesn't do anything if the command is
        a regular command.

        .. note::

            Similar to the :meth:`~discord.app_commands.Command.autocomplete` method, this
            takes :class:`~discord.Interaction` as a parameter rather than a :class:`Context`.

        Parameters
        -----------
        name: :class:`str`
            The parameter name to register as autocomplete.

        Raises
        -------
        TypeError
            The coroutine passed is not actually a coroutine or
            the parameter is not found or of an invalid type.
        """
        ...
    


class HybridGroup(Group[CogT, P, T]):
    r"""A class that is both an application command group and a regular text group.

    This has the same parameters and attributes as a regular :class:`~discord.ext.commands.Group`.
    However, it also doubles as an :class:`application command group <discord.app_commands.Group>`.
    Note that application commands groups cannot have callbacks associated with them, so the callback
    is only called if it's not invoked as an application command.

    Hybrid groups will always have :attr:`Group.invoke_without_command` set to ``True``.

    These are not created manually, instead they are created via the
    decorator or functional interface.

    .. versionadded:: 2.0

    Attributes
    -----------
    fallback: Optional[:class:`str`]
        The command name to use as a fallback for the application command. Since
        application command groups cannot be invoked, this creates a subcommand within
        the group that can be invoked with the given group callback. If ``None``
        then no fallback command is given. Defaults to ``None``.
    fallback_locale: Optional[:class:`~discord.app_commands.locale_str`]
        The fallback command name's locale string, if available.
    """
    __commands_is_hybrid__: ClassVar[bool] = ...
    def __init__(self, *args: Any, name: Union[str, app_commands.locale_str] = ..., description: Union[str, app_commands.locale_str] = ..., fallback: Optional[Union[str, app_commands.locale_str]] = ..., **attrs: Any) -> None:
        ...
    
    @property
    def cog(self) -> CogT:
        ...
    
    @cog.setter
    def cog(self, value: CogT) -> None:
        ...
    
    async def can_run(self, ctx: Context[BotT], /) -> bool:
        ...
    
    def autocomplete(self, name: str) -> Callable[[AutocompleteCallback[CogT, ChoiceT]], AutocompleteCallback[CogT, ChoiceT]]:
        """A decorator that registers a coroutine as an autocomplete prompt for a parameter.

        This is the same as :meth:`~discord.app_commands.Command.autocomplete`. It is only
        applicable for the application command and doesn't do anything if the command is
        a regular command.

        This is only available if the group has a fallback application command registered.

        .. note::

            Similar to the :meth:`~discord.app_commands.Command.autocomplete` method, this
            takes :class:`~discord.Interaction` as a parameter rather than a :class:`Context`.

        Parameters
        -----------
        name: :class:`str`
            The parameter name to register as autocomplete.

        Raises
        -------
        TypeError
            The coroutine passed is not actually a coroutine or
            the parameter is not found or of an invalid type.
        """
        ...
    
    def add_command(self, command: Union[HybridGroup[CogT, ..., Any], HybridCommand[CogT, ..., Any]], /) -> None:
        """Adds a :class:`.HybridCommand` into the internal list of commands.

        This is usually not called, instead the :meth:`~.GroupMixin.command` or
        :meth:`~.GroupMixin.group` shortcut decorators are used instead.

        Parameters
        -----------
        command: :class:`HybridCommand`
            The command to add.

        Raises
        -------
        CommandRegistrationError
            If the command or its alias is already registered by different command.
        TypeError
            If the command passed is not a subclass of :class:`.HybridCommand`.
        """
        ...
    
    def remove_command(self, name: str, /) -> Optional[Command[CogT, ..., Any]]:
        ...
    
    def command(self, name: Union[str, app_commands.locale_str] = ..., *args: Any, with_app_command: bool = ..., **kwargs: Any) -> Callable[[CommandCallback[CogT, ContextT, P2, U]], HybridCommand[CogT, P2, U]]:
        """A shortcut decorator that invokes :func:`~discord.ext.commands.hybrid_command` and adds it to
        the internal command list via :meth:`add_command`.

        Returns
        --------
        Callable[..., :class:`HybridCommand`]
            A decorator that converts the provided method into a Command, adds it to the bot, then returns it.
        """
        ...
    
    def group(self, name: Union[str, app_commands.locale_str] = ..., *args: Any, with_app_command: bool = ..., **kwargs: Any) -> Callable[[CommandCallback[CogT, ContextT, P2, U]], HybridGroup[CogT, P2, U]]:
        """A shortcut decorator that invokes :func:`~discord.ext.commands.hybrid_group` and adds it to
        the internal command list via :meth:`~.GroupMixin.add_command`.

        Returns
        --------
        Callable[..., :class:`HybridGroup`]
            A decorator that converts the provided method into a Group, adds it to the bot, then returns it.
        """
        ...
    


def hybrid_command(name: Union[str, app_commands.locale_str] = ..., *, with_app_command: bool = ..., **attrs: Any) -> Callable[[CommandCallback[CogT, ContextT, P, T]], HybridCommand[CogT, P, T]]:
    r"""A decorator that transforms a function into a :class:`.HybridCommand`.

    A hybrid command is one that functions both as a regular :class:`.Command`
    and one that is also a :class:`app_commands.Command <discord.app_commands.Command>`.

    The callback being attached to the command must be representable as an
    application command callback. Converters are silently converted into a
    :class:`~discord.app_commands.Transformer` with a
    :attr:`discord.AppCommandOptionType.string` type.

    Checks and error handlers are dispatched and called as-if they were commands
    similar to :class:`.Command`. This means that they take :class:`Context` as
    a parameter rather than :class:`discord.Interaction`.

    All checks added using the :func:`.check` & co. decorators are added into
    the function. There is no way to supply your own checks through this
    decorator.

    .. versionadded:: 2.0

    Parameters
    -----------
    name: Union[:class:`str`, :class:`~discord.app_commands.locale_str`]
        The name to create the command with. By default this uses the
        function name unchanged.
    with_app_command: :class:`bool`
        Whether to register the command also as an application command.
    \*\*attrs
        Keyword arguments to pass into the construction of the
        hybrid command.

    Raises
    -------
    TypeError
        If the function is not a coroutine or is already a command.
    """
    ...

def hybrid_group(name: Union[str, app_commands.locale_str] = ..., *, with_app_command: bool = ..., **attrs: Any) -> Callable[[CommandCallback[CogT, ContextT, P, T]], HybridGroup[CogT, P, T]]:
    """A decorator that transforms a function into a :class:`.HybridGroup`.

    This is similar to the :func:`~discord.ext.commands.group` decorator except it creates
    a hybrid group instead.

    Parameters
    -----------
    name: Union[:class:`str`, :class:`~discord.app_commands.locale_str`]
        The name to create the group with. By default this uses the
        function name unchanged.
    with_app_command: :class:`bool`
        Whether to register the command also as an application command.

    Raises
    -------
    TypeError
        If the function is not a coroutine or is already a command.
    """
    ...

