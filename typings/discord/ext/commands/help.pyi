"""
This type stub file was generated by pyright.
"""

import discord.abc
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, TYPE_CHECKING, Tuple, TypeVar
from .core import Command, Group
from .errors import CommandError
from typing_extensions import Self
from .context import Context
from .cog import Cog
from ._types import BotT, UserCheck

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
__all__ = ('Paginator', 'HelpCommand', 'DefaultHelpCommand', 'MinimalHelpCommand')
FuncT = TypeVar('FuncT', bound=Callable[..., Any])
MISSING: Any = ...
class Paginator:
    """A class that aids in paginating code blocks for Discord messages.

    .. container:: operations

        .. describe:: len(x)

            Returns the total number of characters in the paginator.

    Attributes
    -----------
    prefix: Optional[:class:`str`]
        The prefix inserted to every page. e.g. three backticks, if any.
    suffix: Optional[:class:`str`]
        The suffix appended at the end of every page. e.g. three backticks, if any.
    max_size: :class:`int`
        The maximum amount of codepoints allowed in a page.
    linesep: :class:`str`
        The character string inserted between lines. e.g. a newline character.
            .. versionadded:: 1.7
    """
    def __init__(self, prefix: Optional[str] = ..., suffix: Optional[str] = ..., max_size: int = ..., linesep: str = ...) -> None:
        ...
    
    def clear(self) -> None:
        """Clears the paginator to have no pages."""
        ...
    
    def add_line(self, line: str = ..., *, empty: bool = ...) -> None:
        """Adds a line to the current page.

        If the line exceeds the :attr:`max_size` then an exception
        is raised.

        Parameters
        -----------
        line: :class:`str`
            The line to add.
        empty: :class:`bool`
            Indicates if another empty line should be added.

        Raises
        ------
        RuntimeError
            The line was too big for the current :attr:`max_size`.
        """
        ...
    
    def close_page(self) -> None:
        """Prematurely terminate a page."""
        ...
    
    def __len__(self) -> int:
        ...
    
    @property
    def pages(self) -> List[str]:
        """List[:class:`str`]: Returns the rendered list of pages."""
        ...
    
    def __repr__(self) -> str:
        ...
    


class _HelpCommandImpl(Command):
    def __init__(self, inject: HelpCommand, *args: Any, **kwargs: Any) -> None:
        ...
    
    async def prepare(self, ctx: Context[Any]) -> None:
        ...
    
    def update(self, **kwargs: Any) -> None:
        ...
    


class HelpCommand:
    r"""The base implementation for help command formatting.

    .. note::

        Internally instances of this class are deep copied every time
        the command itself is invoked to prevent a race condition
        mentioned in :issue:`2123`.

        This means that relying on the state of this class to be
        the same between command invocations would not work as expected.

    Attributes
    ------------
    context: Optional[:class:`Context`]
        The context that invoked this help formatter. This is generally set after
        the help command assigned, :func:`command_callback`\, has been called.
    show_hidden: :class:`bool`
        Specifies if hidden commands should be shown in the output.
        Defaults to ``False``.
    verify_checks: Optional[:class:`bool`]
        Specifies if commands should have their :attr:`.Command.checks` called
        and verified. If ``True``, always calls :attr:`.Command.checks`.
        If ``None``, only calls :attr:`.Command.checks` in a guild setting.
        If ``False``, never calls :attr:`.Command.checks`. Defaults to ``True``.

        .. versionchanged:: 1.7
    command_attrs: :class:`dict`
        A dictionary of options to pass in for the construction of the help command.
        This allows you to change the command behaviour without actually changing
        the implementation of the command. The attributes will be the same as the
        ones passed in the :class:`.Command` constructor.
    """
    MENTION_TRANSFORMS = ...
    MENTION_PATTERN = ...
    if TYPE_CHECKING:
        __original_kwargs__: Dict[str, Any]
        __original_args__: Tuple[Any, ...]
        ...
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        ...
    
    def __init__(self, **options: Any) -> None:
        ...
    
    def copy(self) -> Self:
        ...
    
    def add_check(self, func: UserCheck[Context[Any]], /) -> None:
        """
        Adds a check to the help command.

        .. versionadded:: 1.4

        .. versionchanged:: 2.0

            ``func`` parameter is now positional-only.

        .. seealso:: The :func:`~discord.ext.commands.check` decorator

        Parameters
        ----------
        func
            The function that will be used as a check.
        """
        ...
    
    def remove_check(self, func: UserCheck[Context[Any]], /) -> None:
        """
        Removes a check from the help command.

        This function is idempotent and will not raise an exception if
        the function is not in the command's checks.

        .. versionadded:: 1.4

        .. versionchanged:: 2.0

            ``func`` parameter is now positional-only.

        Parameters
        ----------
        func
            The function to remove from the checks.
        """
        ...
    
    def get_bot_mapping(self) -> Dict[Optional[Cog], List[Command[Any, ..., Any]]]:
        """Retrieves the bot mapping passed to :meth:`send_bot_help`."""
        ...
    
    @property
    def invoked_with(self) -> Optional[str]:
        """Similar to :attr:`Context.invoked_with` except properly handles
        the case where :meth:`Context.send_help` is used.

        If the help command was used regularly then this returns
        the :attr:`Context.invoked_with` attribute. Otherwise, if
        it the help command was called using :meth:`Context.send_help`
        then it returns the internal command name of the help command.

        Returns
        ---------
        Optional[:class:`str`]
            The command name that triggered this invocation.
        """
        ...
    
    def get_command_signature(self, command: Command[Any, ..., Any], /) -> str:
        """Retrieves the signature portion of the help page.

        .. versionchanged:: 2.0

            ``command`` parameter is now positional-only.

        Parameters
        ------------
        command: :class:`Command`
            The command to get the signature of.

        Returns
        --------
        :class:`str`
            The signature for the command.
        """
        ...
    
    def remove_mentions(self, string: str, /) -> str:
        """Removes mentions from the string to prevent abuse.

        This includes ``@everyone``, ``@here``, member mentions and role mentions.

        .. versionchanged:: 2.0

            ``string`` parameter is now positional-only.

        Returns
        -------
        :class:`str`
            The string with mentions removed.
        """
        ...
    
    @property
    def cog(self) -> Optional[Cog]:
        """A property for retrieving or setting the cog for the help command.

        When a cog is set for the help command, it is as-if the help command
        belongs to that cog. All cog special methods will apply to the help
        command and it will be automatically unset on unload.

        To unbind the cog from the help command, you can set it to ``None``.

        Returns
        --------
        Optional[:class:`Cog`]
            The cog that is currently set for the help command.
        """
        ...
    
    @cog.setter
    def cog(self, cog: Optional[Cog]) -> None:
        ...
    
    def command_not_found(self, string: str, /) -> str:
        """|maybecoro|

        A method called when a command is not found in the help command.
        This is useful to override for i18n.

        Defaults to ``No command called {0} found.``

        .. versionchanged:: 2.0

            ``string`` parameter is now positional-only.

        Parameters
        ------------
        string: :class:`str`
            The string that contains the invalid command. Note that this has
            had mentions removed to prevent abuse.

        Returns
        ---------
        :class:`str`
            The string to use when a command has not been found.
        """
        ...
    
    def subcommand_not_found(self, command: Command[Any, ..., Any], string: str, /) -> str:
        """|maybecoro|

        A method called when a command did not have a subcommand requested in the help command.
        This is useful to override for i18n.

        Defaults to either:

        - ``'Command "{command.qualified_name}" has no subcommands.'``
            - If there is no subcommand in the ``command`` parameter.
        - ``'Command "{command.qualified_name}" has no subcommand named {string}'``
            - If the ``command`` parameter has subcommands but not one named ``string``.

        .. versionchanged:: 2.0

            ``command`` and ``string`` parameters are now positional-only.

        Parameters
        ------------
        command: :class:`Command`
            The command that did not have the subcommand requested.
        string: :class:`str`
            The string that contains the invalid subcommand. Note that this has
            had mentions removed to prevent abuse.

        Returns
        ---------
        :class:`str`
            The string to use when the command did not have the subcommand requested.
        """
        ...
    
    async def filter_commands(self, commands: Iterable[Command[Any, ..., Any]], /, *, sort: bool = ..., key: Optional[Callable[[Command[Any, ..., Any]], Any]] = ...) -> List[Command[Any, ..., Any]]:
        """|coro|

        Returns a filtered list of commands and optionally sorts them.

        This takes into account the :attr:`verify_checks` and :attr:`show_hidden`
        attributes.

        .. versionchanged:: 2.0

            ``commands`` parameter is now positional-only.

        Parameters
        ------------
        commands: Iterable[:class:`Command`]
            An iterable of commands that are getting filtered.
        sort: :class:`bool`
            Whether to sort the result.
        key: Optional[Callable[[:class:`Command`], Any]]
            An optional key function to pass to :func:`py:sorted` that
            takes a :class:`Command` as its sole parameter. If ``sort`` is
            passed as ``True`` then this will default as the command name.

        Returns
        ---------
        List[:class:`Command`]
            A list of commands that passed the filter.
        """
        ...
    
    def get_max_size(self, commands: Sequence[Command[Any, ..., Any]], /) -> int:
        """Returns the largest name length of the specified command list.

        .. versionchanged:: 2.0

            ``commands`` parameter is now positional-only.

        Parameters
        ------------
        commands: Sequence[:class:`Command`]
            A sequence of commands to check for the largest size.

        Returns
        --------
        :class:`int`
            The maximum width of the commands.
        """
        ...
    
    def get_destination(self) -> discord.abc.MessageableChannel:
        """Returns the :class:`~discord.abc.Messageable` where the help command will be output.

        You can override this method to customise the behaviour.

        By default this returns the context's channel.

        Returns
        -------
        :class:`.abc.Messageable`
            The destination where the help command will be output.
        """
        ...
    
    async def send_error_message(self, error: str, /) -> None:
        """|coro|

        Handles the implementation when an error happens in the help command.
        For example, the result of :meth:`command_not_found` will be passed here.

        You can override this method to customise the behaviour.

        By default, this sends the error message to the destination
        specified by :meth:`get_destination`.

        .. note::

            You can access the invocation context with :attr:`HelpCommand.context`.

        .. versionchanged:: 2.0

            ``error`` parameter is now positional-only.

        Parameters
        ------------
        error: :class:`str`
            The error message to display to the user. Note that this has
            had mentions removed to prevent abuse.
        """
        ...
    
    @_not_overridden
    async def on_help_command_error(self, ctx: Context[BotT], error: CommandError, /) -> None:
        """|coro|

        The help command's error handler, as specified by :ref:`ext_commands_error_handler`.

        Useful to override if you need some specific behaviour when the error handler
        is called.

        By default this method does nothing and just propagates to the default
        error handlers.

        .. versionchanged:: 2.0

            ``ctx`` and ``error`` parameters are now positional-only.

        Parameters
        ------------
        ctx: :class:`Context`
            The invocation context.
        error: :class:`CommandError`
            The error that was raised.
        """
        ...
    
    async def send_bot_help(self, mapping: Mapping[Optional[Cog], List[Command[Any, ..., Any]]], /) -> None:
        """|coro|

        Handles the implementation of the bot command page in the help command.
        This function is called when the help command is called with no arguments.

        It should be noted that this method does not return anything -- rather the
        actual message sending should be done inside this method. Well behaved subclasses
        should use :meth:`get_destination` to know where to send, as this is a customisation
        point for other users.

        You can override this method to customise the behaviour.

        .. note::

            You can access the invocation context with :attr:`HelpCommand.context`.

            Also, the commands in the mapping are not filtered. To do the filtering
            you will have to call :meth:`filter_commands` yourself.

        .. versionchanged:: 2.0

            ``mapping`` parameter is now positional-only.

        Parameters
        ------------
        mapping: Mapping[Optional[:class:`Cog`], List[:class:`Command`]]
            A mapping of cogs to commands that have been requested by the user for help.
            The key of the mapping is the :class:`~.commands.Cog` that the command belongs to, or
            ``None`` if there isn't one, and the value is a list of commands that belongs to that cog.
        """
        ...
    
    async def send_cog_help(self, cog: Cog, /) -> None:
        """|coro|

        Handles the implementation of the cog page in the help command.
        This function is called when the help command is called with a cog as the argument.

        It should be noted that this method does not return anything -- rather the
        actual message sending should be done inside this method. Well behaved subclasses
        should use :meth:`get_destination` to know where to send, as this is a customisation
        point for other users.

        You can override this method to customise the behaviour.

        .. note::

            You can access the invocation context with :attr:`HelpCommand.context`.

            To get the commands that belong to this cog see :meth:`Cog.get_commands`.
            The commands returned not filtered. To do the filtering you will have to call
            :meth:`filter_commands` yourself.

        .. versionchanged:: 2.0

            ``cog`` parameter is now positional-only.

        Parameters
        -----------
        cog: :class:`Cog`
            The cog that was requested for help.
        """
        ...
    
    async def send_group_help(self, group: Group[Any, ..., Any], /) -> None:
        """|coro|

        Handles the implementation of the group page in the help command.
        This function is called when the help command is called with a group as the argument.

        It should be noted that this method does not return anything -- rather the
        actual message sending should be done inside this method. Well behaved subclasses
        should use :meth:`get_destination` to know where to send, as this is a customisation
        point for other users.

        You can override this method to customise the behaviour.

        .. note::

            You can access the invocation context with :attr:`HelpCommand.context`.

            To get the commands that belong to this group without aliases see
            :attr:`Group.commands`. The commands returned not filtered. To do the
            filtering you will have to call :meth:`filter_commands` yourself.

        .. versionchanged:: 2.0

            ``group`` parameter is now positional-only.

        Parameters
        -----------
        group: :class:`Group`
            The group that was requested for help.
        """
        ...
    
    async def send_command_help(self, command: Command[Any, ..., Any], /) -> None:
        """|coro|

        Handles the implementation of the single command page in the help command.

        It should be noted that this method does not return anything -- rather the
        actual message sending should be done inside this method. Well behaved subclasses
        should use :meth:`get_destination` to know where to send, as this is a customisation
        point for other users.

        You can override this method to customise the behaviour.

        .. note::

            You can access the invocation context with :attr:`HelpCommand.context`.

        .. admonition:: Showing Help
            :class: helpful

            There are certain attributes and methods that are helpful for a help command
            to show such as the following:

            - :attr:`Command.help`
            - :attr:`Command.brief`
            - :attr:`Command.short_doc`
            - :attr:`Command.description`
            - :meth:`get_command_signature`

            There are more than just these attributes but feel free to play around with
            these to help you get started to get the output that you want.

        .. versionchanged:: 2.0

            ``command`` parameter is now positional-only.

        Parameters
        -----------
        command: :class:`Command`
            The command that was requested for help.
        """
        ...
    
    async def prepare_help_command(self, ctx: Context[BotT], command: Optional[str] = ..., /) -> None:
        """|coro|

        A low level method that can be used to prepare the help command
        before it does anything. For example, if you need to prepare
        some state in your subclass before the command does its processing
        then this would be the place to do it.

        The default implementation does nothing.

        .. note::

            This is called *inside* the help command callback body. So all
            the usual rules that happen inside apply here as well.

        .. versionchanged:: 2.0

            ``ctx`` and ``command`` parameters are now positional-only.

        Parameters
        -----------
        ctx: :class:`Context`
            The invocation context.
        command: Optional[:class:`str`]
            The argument passed to the help command.
        """
        ...
    
    async def command_callback(self, ctx: Context[BotT], /, *, command: Optional[str] = ...) -> None:
        """|coro|

        The actual implementation of the help command.

        It is not recommended to override this method and instead change
        the behaviour through the methods that actually get dispatched.

        - :meth:`send_bot_help`
        - :meth:`send_cog_help`
        - :meth:`send_group_help`
        - :meth:`send_command_help`
        - :meth:`get_destination`
        - :meth:`command_not_found`
        - :meth:`subcommand_not_found`
        - :meth:`send_error_message`
        - :meth:`on_help_command_error`
        - :meth:`prepare_help_command`

        .. versionchanged:: 2.0

            ``ctx`` parameter is now positional-only.
        """
        ...
    


class DefaultHelpCommand(HelpCommand):
    """The implementation of the default help command.

    This inherits from :class:`HelpCommand`.

    It extends it with the following attributes.

    Attributes
    ------------
    width: :class:`int`
        The maximum number of characters that fit in a line.
        Defaults to 80.
    sort_commands: :class:`bool`
        Whether to sort the commands in the output alphabetically. Defaults to ``True``.
    dm_help: Optional[:class:`bool`]
        A tribool that indicates if the help command should DM the user instead of
        sending it to the channel it received it from. If the boolean is set to
        ``True``, then all help output is DM'd. If ``False``, none of the help
        output is DM'd. If ``None``, then the bot will only DM when the help
        message becomes too long (dictated by more than :attr:`dm_help_threshold` characters).
        Defaults to ``False``.
    dm_help_threshold: Optional[:class:`int`]
        The number of characters the paginator must accumulate before getting DM'd to the
        user if :attr:`dm_help` is set to ``None``. Defaults to 1000.
    indent: :class:`int`
        How much to indent the commands from a heading. Defaults to ``2``.
    arguments_heading: :class:`str`
        The arguments list's heading string used when the help command is invoked with a command name.
        Useful for i18n. Defaults to ``"Arguments:"``.
        Shown when :attr:`.show_parameter_descriptions` is ``True``.

        .. versionadded:: 2.0
    show_parameter_descriptions: :class:`bool`
        Whether to show the parameter descriptions. Defaults to ``True``.
        Setting this to ``False`` will revert to showing the :attr:`~.commands.Command.signature` instead.

        .. versionadded:: 2.0
    commands_heading: :class:`str`
        The command list's heading string used when the help command is invoked with a category name.
        Useful for i18n. Defaults to ``"Commands:"``
    default_argument_description: :class:`str`
        The default argument description string used when the argument's :attr:`~.commands.Parameter.description` is ``None``.
        Useful for i18n. Defaults to ``"No description given."``

        .. versionadded:: 2.0
    no_category: :class:`str`
        The string used when there is a command which does not belong to any category(cog).
        Useful for i18n. Defaults to ``"No Category"``
    paginator: :class:`Paginator`
        The paginator used to paginate the help command output.
    """
    def __init__(self, **options: Any) -> None:
        ...
    
    def shorten_text(self, text: str, /) -> str:
        """:class:`str`: Shortens text to fit into the :attr:`width`.

        .. versionchanged:: 2.0

            ``text`` parameter is now positional-only.
        """
        ...
    
    def get_ending_note(self) -> str:
        """:class:`str`: Returns help command's ending note. This is mainly useful to override for i18n purposes."""
        ...
    
    def get_command_signature(self, command: Command[Any, ..., Any], /) -> str:
        """Retrieves the signature portion of the help page.

        Calls :meth:`~.HelpCommand.get_command_signature` if :attr:`show_parameter_descriptions` is ``False``
        else returns a modified signature where the command parameters are not shown.

        .. versionadded:: 2.0

        Parameters
        ------------
        command: :class:`Command`
            The command to get the signature of.

        Returns
        --------
        :class:`str`
            The signature for the command.
        """
        ...
    
    def add_indented_commands(self, commands: Sequence[Command[Any, ..., Any]], /, *, heading: str, max_size: Optional[int] = ...) -> None:
        """Indents a list of commands after the specified heading.

        The formatting is added to the :attr:`paginator`.

        The default implementation is the command name indented by
        :attr:`indent` spaces, padded to ``max_size`` followed by
        the command's :attr:`Command.short_doc` and then shortened
        to fit into the :attr:`width`.

        .. versionchanged:: 2.0
            ``commands`` parameter is now positional-only.

        Parameters
        -----------
        commands: Sequence[:class:`Command`]
            A list of commands to indent for output.
        heading: :class:`str`
            The heading to add to the output. This is only added
            if the list of commands is greater than 0.
        max_size: Optional[:class:`int`]
            The max size to use for the gap between indents.
            If unspecified, calls :meth:`~HelpCommand.get_max_size` on the
            commands parameter.
        """
        ...
    
    def add_command_arguments(self, command: Command[Any, ..., Any], /) -> None:
        """Indents a list of command arguments after the :attr:`.arguments_heading`.

        The default implementation is the argument :attr:`~.commands.Parameter.name` indented by
        :attr:`indent` spaces, padded to ``max_size`` using :meth:`~HelpCommand.get_max_size`
        followed by the argument's :attr:`~.commands.Parameter.description` or
        :attr:`.default_argument_description` and then shortened
        to fit into the :attr:`width` and then :attr:`~.commands.Parameter.displayed_default`
        between () if one is present after that.

        .. versionadded:: 2.0

        Parameters
        -----------
        command: :class:`Command`
            The command to list the arguments for.
        """
        ...
    
    async def send_pages(self) -> None:
        """|coro|

        A helper utility to send the page output from :attr:`paginator` to the destination.
        """
        ...
    
    def add_command_formatting(self, command: Command[Any, ..., Any], /) -> None:
        """A utility function to format the non-indented block of commands and groups.

        .. versionchanged:: 2.0

            ``command`` parameter is now positional-only.

        .. versionchanged:: 2.0
            :meth:`.add_command_arguments` is now called if :attr:`.show_parameter_descriptions` is ``True``.

        Parameters
        ------------
        command: :class:`Command`
            The command to format.
        """
        ...
    
    def get_destination(self) -> discord.abc.Messageable:
        ...
    
    async def prepare_help_command(self, ctx: Context[BotT], command: Optional[str], /) -> None:
        ...
    
    async def send_bot_help(self, mapping: Mapping[Optional[Cog], List[Command[Any, ..., Any]]], /) -> None:
        ...
    
    async def send_command_help(self, command: Command[Any, ..., Any], /) -> None:
        ...
    
    async def send_group_help(self, group: Group[Any, ..., Any], /) -> None:
        ...
    
    async def send_cog_help(self, cog: Cog, /) -> None:
        ...
    


class MinimalHelpCommand(HelpCommand):
    """An implementation of a help command with minimal output.

    This inherits from :class:`HelpCommand`.

    Attributes
    ------------
    sort_commands: :class:`bool`
        Whether to sort the commands in the output alphabetically. Defaults to ``True``.
    commands_heading: :class:`str`
        The command list's heading string used when the help command is invoked with a category name.
        Useful for i18n. Defaults to ``"Commands"``
    aliases_heading: :class:`str`
        The alias list's heading string used to list the aliases of the command. Useful for i18n.
        Defaults to ``"Aliases:"``.
    dm_help: Optional[:class:`bool`]
        A tribool that indicates if the help command should DM the user instead of
        sending it to the channel it received it from. If the boolean is set to
        ``True``, then all help output is DM'd. If ``False``, none of the help
        output is DM'd. If ``None``, then the bot will only DM when the help
        message becomes too long (dictated by more than :attr:`dm_help_threshold` characters).
        Defaults to ``False``.
    dm_help_threshold: Optional[:class:`int`]
        The number of characters the paginator must accumulate before getting DM'd to the
        user if :attr:`dm_help` is set to ``None``. Defaults to 1000.
    no_category: :class:`str`
        The string used when there is a command which does not belong to any category(cog).
        Useful for i18n. Defaults to ``"No Category"``
    paginator: :class:`Paginator`
        The paginator used to paginate the help command output.
    """
    def __init__(self, **options: Any) -> None:
        ...
    
    async def send_pages(self) -> None:
        """|coro|

        A helper utility to send the page output from :attr:`paginator` to the destination.
        """
        ...
    
    def get_opening_note(self) -> str:
        """Returns help command's opening note. This is mainly useful to override for i18n purposes.

        The default implementation returns ::

            Use `{prefix}{command_name} [command]` for more info on a command.
            You can also use `{prefix}{command_name} [category]` for more info on a category.

        Returns
        -------
        :class:`str`
            The help command opening note.
        """
        ...
    
    def get_command_signature(self, command: Command[Any, ..., Any], /) -> str:
        ...
    
    def get_ending_note(self) -> str:
        """Return the help command's ending note. This is mainly useful to override for i18n purposes.

        The default implementation does nothing.

        Returns
        -------
        :class:`str`
            The help command ending note.
        """
        ...
    
    def add_bot_commands_formatting(self, commands: Sequence[Command[Any, ..., Any]], heading: str, /) -> None:
        """Adds the minified bot heading with commands to the output.

        The formatting should be added to the :attr:`paginator`.

        The default implementation is a bold underline heading followed
        by commands separated by an EN SPACE (U+2002) in the next line.

        .. versionchanged:: 2.0

            ``commands`` and ``heading`` parameters are now positional-only.

        Parameters
        -----------
        commands: Sequence[:class:`Command`]
            A list of commands that belong to the heading.
        heading: :class:`str`
            The heading to add to the line.
        """
        ...
    
    def add_subcommand_formatting(self, command: Command[Any, ..., Any], /) -> None:
        """Adds formatting information on a subcommand.

        The formatting should be added to the :attr:`paginator`.

        The default implementation is the prefix and the :attr:`Command.qualified_name`
        optionally followed by an En dash and the command's :attr:`Command.short_doc`.

        .. versionchanged:: 2.0

            ``command`` parameter is now positional-only.

        Parameters
        -----------
        command: :class:`Command`
            The command to show information of.
        """
        ...
    
    def add_aliases_formatting(self, aliases: Sequence[str], /) -> None:
        """Adds the formatting information on a command's aliases.

        The formatting should be added to the :attr:`paginator`.

        The default implementation is the :attr:`aliases_heading` bolded
        followed by a comma separated list of aliases.

        This is not called if there are no aliases to format.

        .. versionchanged:: 2.0

            ``aliases`` parameter is now positional-only.

        Parameters
        -----------
        aliases: Sequence[:class:`str`]
            A list of aliases to format.
        """
        ...
    
    def add_command_formatting(self, command: Command[Any, ..., Any], /) -> None:
        """A utility function to format commands and groups.

        .. versionchanged:: 2.0

            ``command`` parameter is now positional-only.

        Parameters
        ------------
        command: :class:`Command`
            The command to format.
        """
        ...
    
    def get_destination(self) -> discord.abc.Messageable:
        ...
    
    async def prepare_help_command(self, ctx: Context[BotT], command: Optional[str], /) -> None:
        ...
    
    async def send_bot_help(self, mapping: Mapping[Optional[Cog], List[Command[Any, ..., Any]]], /) -> None:
        ...
    
    async def send_cog_help(self, cog: Cog, /) -> None:
        ...
    
    async def send_group_help(self, group: Group[Any, ..., Any], /) -> None:
        ...
    
    async def send_command_help(self, command: Command[Any, ..., Any], /) -> None:
        ...
    


