"""Base context implementation for Discord commands.

This module contains the base context class used for Discord command handling.
"""
from __future__ import annotations

import io

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Generic, Optional, Protocol, TypeVar, Union

import discord

from discord.ext import commands


if TYPE_CHECKING:
    from types import TracebackType

    from aiohttp import ClientSession
    from redis.asyncio import ConnectionPool as RedisConnectionPool

T = TypeVar("T")

class ConnectionContextManager(Protocol):
    """Protocol defining the connection context manager interface."""

    async def __aenter__(self) -> RedisConnectionPool: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class ConfirmationView(discord.ui.View):
    """View for confirmation dialogs."""

    def __init__(self, *, timeout: float, author_id: int, delete_after: bool) -> None:
        """Initialize the confirmation view.

        Args:
            timeout: How long to wait before timing out
            author_id: ID of the user who can interact with this view
            delete_after: Whether to delete the message after confirmation
        """
        super().__init__(timeout=timeout)
        self.value: bool | None = None
        self.delete_after: bool = delete_after
        self.author_id: int = author_id
        self.message: discord.Message | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Check if the interaction is from the authorized user.

        Args:
            interaction: The interaction to check

        Returns:
            bool: Whether the interaction is valid
        """
        if interaction.user and interaction.user.id == self.author_id:
            return True
        await interaction.response.send_message("This confirmation dialog is not for you.", ephemeral=True)
        return False

    async def on_timeout(self) -> None:
        """Handle view timeout."""
        if self.delete_after and self.message:
            await self.message.delete()

    @discord.ui.button(label="Confirm", style=discord.ButtonStyle.green)
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handle confirmation button click.

        Args:
            interaction: The button interaction
            button: The button that was clicked
        """
        self.value = True
        await interaction.response.defer()
        if self.delete_after:
            await interaction.delete_original_response()
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.red)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Handle cancellation button click.

        Args:
            interaction: The button interaction
            button: The button that was clicked
        """
        self.value = False
        await interaction.response.defer()
        if self.delete_after:
            await interaction.delete_original_response()
        self.stop()


class DisambiguatorView(discord.ui.View, Generic[T]):
    """View for disambiguation selection menus."""

    message: discord.Message
    selected: T

    def __init__(self, ctx: BaseContext, data: list[T], entry: Callable[[T], Any]):
        """Initialize the disambiguator view.

        Args:
            ctx: The command context
            data: List of items to disambiguate
            entry: Function to convert items to select options
        """
        super().__init__()
        self.ctx: BaseContext = ctx
        self.data: list[T] = data

        options = []
        for i, x in enumerate(data):
            opt = entry(x)
            if not isinstance(opt, discord.SelectOption):
                opt = discord.SelectOption(label=str(opt))
            opt.value = str(i)
            options.append(opt)

        select = discord.ui.Select(options=options)
        select.callback = self.on_select_submit
        self.select = select
        self.add_item(select)

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        """Check if the interaction is from the authorized user.

        Args:
            interaction: The interaction to check

        Returns:
            bool: Whether the interaction is valid
        """
        if interaction.user.id != self.ctx.author.id:
            await interaction.response.send_message("This select menu is not meant for you, sorry.", ephemeral=True)
            return False
        return True

    async def on_select_submit(self, interaction: discord.Interaction):
        """Handle select menu submission.

        Args:
            interaction: The select menu interaction
        """
        index = int(self.select.values[0])
        self.selected = self.data[index]
        await interaction.response.defer()
        if not self.message.flags.ephemeral:
            await self.message.delete()
        self.stop()


class BaseContext(commands.Context):
    """Base context class for Discord commands."""

    channel: discord.VoiceChannel | discord.TextChannel | discord.Thread | discord.DMChannel
    prefix: str
    command: commands.Command[Any, ..., Any]

    def __init__(self, **kwargs):
        """Initialize the base context.

        Args:
            **kwargs: Keyword arguments to pass to the parent class
        """
        super().__init__(**kwargs)
        self.pool: RedisConnectionPool | None = None

    async def entry_to_code(self, entries: Iterable[tuple[str, str]]) -> None:
        """Convert entries to a code block and send it.

        Args:
            entries: Iterable of name-value pairs to format
        """
        width = max(len(a) for a, b in entries)
        output = ["```"]
        output.extend(f"{name:<{width}}: {entry}" for name, entry in entries)
        output.append("```")
        await self.send("\n".join(output))

    async def indented_entry_to_code(self, entries: Iterable[tuple[str, str]]) -> None:
        """Convert entries to an indented code block and send it.

        Args:
            entries: Iterable of name-value pairs to format
        """
        width = max(len(a) for a, b in entries)
        output = ["```"]
        output.extend(f"\u200b{name:>{width}}: {entry}" for name, entry in entries)
        output.append("```")
        await self.send("\n".join(output))

    def __repr__(self) -> str:
        """Get string representation of the context.

        Returns:
            str: String representation
        """
        return "<Context>"

    @property
    def session(self) -> ClientSession:
        """Get the client session.

        Returns:
            ClientSession: The client session
        """
        return self.bot.session

    @discord.utils.cached_property
    def replied_reference(self) -> discord.MessageReference | None:
        """Get the message reference for a reply.

        Returns:
            Optional[MessageReference]: The message reference if this is a reply
        """
        ref = self.message.reference
        if ref and isinstance(ref.resolved, discord.Message):
            return ref.resolved.to_reference()
        return None

    @discord.utils.cached_property
    def replied_message(self) -> discord.Message | None:
        """Get the message being replied to.

        Returns:
            Optional[Message]: The message if this is a reply
        """
        ref = self.message.reference
        if ref and isinstance(ref.resolved, discord.Message):
            return ref.resolved
        return None

    async def disambiguate(self, matches: list[T], entry: Callable[[T], Any], *, ephemeral: bool = False) -> T:
        """Disambiguate between multiple matches.

        Args:
            matches: List of items to disambiguate between
            entry: Function to convert items to select options
            ephemeral: Whether the disambiguation message should be ephemeral

        Returns:
            T: The selected item

        Raises:
            ValueError: If there are no matches or too many matches
        """
        if not matches:
            raise ValueError("No results found.")

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 25:
            raise ValueError("Too many results... sorry.")

        view = DisambiguatorView(self, matches, entry)
        view.message = await self.send(
            "There are too many matches... Which one did you mean?",
            view=view,
            ephemeral=ephemeral,
        )
        await view.wait()
        return view.selected

    async def prompt(
        self,
        message: str,
        *,
        timeout: float = 60.0,
        delete_after: bool = True,
        author_id: int | None = None,
    ) -> bool | None:
        """An interactive reaction confirmation dialog.

        Args:
            message: The message to show along with the prompt
            timeout: How long to wait before returning
            delete_after: Whether to delete the confirmation message after we're done
            author_id: The member who should respond to the prompt. Defaults to the author of the
                Context's message

        Returns:
            Optional[bool]: True if explicit confirm, False if explicit deny, None if deny due to timeout
        """
        author_id = author_id or self.author.id
        view = ConfirmationView(
            timeout=timeout,
            delete_after=delete_after,
            author_id=author_id,
        )
        view.message = await self.send(message, view=view, ephemeral=delete_after)
        await view.wait()
        return view.value

    def tick(self, opt: bool | None, label: str | None = None) -> str:
        """Get a tick emoji based on a boolean value.

        Args:
            opt: The boolean value to convert to an emoji
            label: Optional label to append after the emoji

        Returns:
            str: The emoji string
        """
        lookup = {
            True: "<:greenTick:330090705336664065>",
            False: "<:redTick:330090723011592193>",
            None: "<:greyTick:563231201280917524>",
        }
        emoji = lookup.get(opt, "<:redTick:330090723011592193>")
        return f"{emoji}: {label}" if label is not None else emoji

    @property
    def db(self) -> RedisConnectionPool:
        """Get the database connection pool.

        Returns:
            RedisConnectionPool: The database connection pool
        """
        return self.pool

    async def show_help(self, command: Any = None) -> None:
        """Show the help command for the specified command if given.

        If no command is given, then it'll show help for the current command.

        Args:
            command: The command to show help for
        """
        cmd = self.bot.get_command("help")
        if cmd is None:
            return

        if command is None:
            command = self.command.qualified_name if self.command else None

        await self.invoke(cmd, command=command)
