"""
This type stub file was generated by pyright.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from re import Pattern
from typing import TYPE_CHECKING, Any, Dict, List, Set, Tuple, Type

from typing_extensions import Self, TypeGuard

from ._types import BotT
from .context import Context

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
__all__ = ('Flag', 'flag', 'FlagConverter')
if TYPE_CHECKING:
    ...
@dataclass
class Flag:
    """Represents a flag parameter for :class:`FlagConverter`.

    The :func:`~discord.ext.commands.flag` function helps
    create these flag objects, but it is not necessary to
    do so. These cannot be constructed manually.

    Attributes
    ------------
    name: :class:`str`
        The name of the flag.
    aliases: List[:class:`str`]
        The aliases of the flag name.
    attribute: :class:`str`
        The attribute in the class that corresponds to this flag.
    default: Any
        The default value of the flag, if available.
    annotation: Any
        The underlying evaluated annotation of the flag.
    max_args: :class:`int`
        The maximum number of arguments the flag can accept.
        A negative value indicates an unlimited amount of arguments.
    override: :class:`bool`
        Whether multiple given values overrides the previous value.
    description: :class:`str`
        The description of the flag. Shown for hybrid commands when they're
        used as application commands.
    """
    name: str = ...
    aliases: List[str] = ...
    attribute: str = ...
    annotation: Any = ...
    default: Any = ...
    max_args: int = ...
    override: bool = ...
    description: str = ...
    cast_to_dict: bool = ...
    @property
    def required(self) -> bool:
        """:class:`bool`: Whether the flag is required.

        A required flag has no default value.
        """
        ...



def flag(*, name: str = ..., aliases: List[str] = ..., default: Any = ..., max_args: int = ..., override: bool = ..., converter: Any = ..., description: str = ...) -> Any:
    """Override default functionality and parameters of the underlying :class:`FlagConverter`
    class attributes.

    Parameters
    ------------
    name: :class:`str`
        The flag name. If not given, defaults to the attribute name.
    aliases: List[:class:`str`]
        Aliases to the flag name. If not given no aliases are set.
    default: Any
        The default parameter. This could be either a value or a callable that takes
        :class:`Context` as its sole parameter. If not given then it defaults to
        the default value given to the attribute.
    max_args: :class:`int`
        The maximum number of arguments the flag can accept.
        A negative value indicates an unlimited amount of arguments.
        The default value depends on the annotation given.
    override: :class:`bool`
        Whether multiple given values overrides the previous value. The default
        value depends on the annotation given.
    converter: Any
        The converter to use for this flag. This replaces the annotation at
        runtime which is transparent to type checkers.
    description: :class:`str`
        The description of the flag. Shown for hybrid commands when they're
        used as application commands.
    """
    ...

def is_flag(obj: Any) -> TypeGuard[Type[FlagConverter]]:
    ...

def validate_flag_name(name: str, forbidden: Set[str]) -> None:
    ...

def get_flags(namespace: Dict[str, Any], globals: Dict[str, Any], locals: Dict[str, Any]) -> Dict[str, Flag]:
    ...

class FlagsMeta(type):
    if TYPE_CHECKING:
        __commands_is_flag__: bool
        __commands_flags__: Dict[str, Flag]
        __commands_flag_aliases__: Dict[str, str]
        __commands_flag_regex__: Pattern[str]
        __commands_flag_case_insensitive__: bool
        __commands_flag_delimiter__: str
        __commands_flag_prefix__: str
        ...
    def __new__(cls, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any], *, case_insensitive: bool = ..., delimiter: str = ..., prefix: str = ...) -> Self:
        ...



async def tuple_convert_all(ctx: Context[BotT], argument: str, flag: Flag, converter: Any) -> Tuple[Any, ...]:
    ...

async def tuple_convert_flag(ctx: Context[BotT], argument: str, flag: Flag, converters: Any) -> Tuple[Any, ...]:
    ...

async def convert_flag(ctx: Context[BotT], argument: str, flag: Flag, annotation: Any = ...) -> Any:
    ...

class FlagConverter(metaclass=FlagsMeta):
    """A converter that allows for a user-friendly flag syntax.

    The flags are defined using :pep:`526` type annotations similar
    to the :mod:`dataclasses` Python module. For more information on
    how this converter works, check the appropriate
    :ref:`documentation <ext_commands_flag_converter>`.

    .. container:: operations

        .. describe:: iter(x)

            Returns an iterator of ``(flag_name, flag_value)`` pairs. This allows it
            to be, for example, constructed as a dict or a list of pairs.
            Note that aliases are not shown.

    .. versionadded:: 2.0

    Parameters
    -----------
    case_insensitive: :class:`bool`
        A class parameter to toggle case insensitivity of the flag parsing.
        If ``True`` then flags are parsed in a case insensitive manner.
        Defaults to ``False``.
    prefix: :class:`str`
        The prefix that all flags must be prefixed with. By default
        there is no prefix.
    delimiter: :class:`str`
        The delimiter that separates a flag's argument from the flag's name.
        By default this is ``:``.
    """
    @classmethod
    def get_flags(cls) -> Dict[str, Flag]:
        """Dict[:class:`str`, :class:`Flag`]: A mapping of flag name to flag object this converter has."""
        ...

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        ...

    def __repr__(self) -> str:
        ...

    @classmethod
    def parse_flags(cls, argument: str, *, ignore_extra: bool = ...) -> Dict[str, List[str]]:
        ...

    @classmethod
    async def convert(cls, ctx: Context[BotT], argument: str) -> Self:
        """|coro|

        The method that actually converters an argument to the flag mapping.

        Parameters
        ----------
        ctx: :class:`Context`
            The invocation context.
        argument: :class:`str`
            The argument to convert from.

        Raises
        --------
        FlagError
            A flag related parsing error.

        Returns
        --------
        :class:`FlagConverter`
            The flag converter instance with all flags parsed.
        """
        ...
