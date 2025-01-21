# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false

"""Core DemocracyBot implementation.

This module contains the main DemocracyBot class and its core functionality.

Notes:
    Important implementation details:
    - Resource management: Uses ResourceManager for memory and task limits
    - Extension loading: Follows dependency order with retries
    - Message handling: Includes size validation and memory checks
    - Error handling: Comprehensive error capture and logging
    - Cleanup: Proper resource cleanup with timeouts
    - Rate limiting: Configurable rate limits with spam protection

Missing or needs improvement:
    - More detailed error handling patterns
    - Enhanced validation frameworks
    - Comprehensive test coverage
    - Additional documentation
"""
from __future__ import annotations

import asyncio
import datetime
import gc

from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple, TypeVar, Union, cast

import aiohttp
import discord
import pysnooper
import structlog

from discord import (
    Activity,
    AllowedMentions,
    AppInfo,
    DMChannel,
    Game,
    Guild,
    Intents,
    Message,
    Status,
    TextChannel,
    Thread,
    User,
)
from discord.abc import Messageable
from discord.ext import commands
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph  # type: ignore[import]


logger = structlog.get_logger(__name__)
from pydantic import SecretStr

import democracy_exe

from democracy_exe import utils
from democracy_exe.agentic.workflows.react.graph import graph as memgraph
from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.handlers.attachment_handler import AttachmentHandler
from democracy_exe.chatbot.handlers.message_handler import MessageHandler
from democracy_exe.chatbot.utils.extension_manager import get_extension_load_order, load_extension_with_retry
from democracy_exe.chatbot.utils.guild_utils import preload_guild_data
from democracy_exe.chatbot.utils.message_utils import format_inbound_message
from democracy_exe.chatbot.utils.resource_manager import ResourceLimits, ResourceManager
from democracy_exe.constants import CHANNEL_ID
from democracy_exe.utils.bot_context import Context


if TYPE_CHECKING:
    from redis.asyncio import ConnectionPool as RedisConnectionPool

DESCRIPTION = """An example bot to showcase the discord.ext.commands extension
module.

There are a number of utility commands being showcased here."""

class ProxyObject(discord.Object):
    def __init__(self, guild: discord.abc.Snowflake | None):
        super().__init__(id=0)
        self.guild: discord.abc.Snowflake | None = guild

def _prefix_callable(bot: DemocracyBot, msg: discord.Message) -> list[str]:
    """
    Generate a list of command prefixes for the bot.

    This function generates a list of command prefixes for the bot based on the message context.
    If the message is from a direct message (DM) channel, it includes the bot's user ID mentions
    and default prefixes. If the message is from a guild (server) channel, it includes the bot's
    user ID mentions and the guild-specific prefixes.

    Args:
    ----
        bot (AsyncGoobBot): The instance of the bot.
        msg (discord.Message): The message object from Discord.

    Returns:
    -------
        List[str]: A list of command prefixes to be used for the bot.

    """
    user_id = bot.user.id
    base = [f"<@!{user_id}> ", f"<@{user_id}> "]
    if msg.guild is None:  # pyright: ignore[reportAttributeAccessIssue]
        base.extend(("!", "?"))
    else:
        base.extend(bot.prefixes.get(msg.guild.id, ["?", "!"]))  # pyright: ignore[reportAttributeAccessIssue]
    return base

def get_extensions() -> list[str]:
    """Get list of available extensions.

    Returns:
        list[str]: List of extension module paths
    """
    from democracy_exe.chatbot.utils.extension_manager import extensions
    return list(extensions())

class DemocracyBot(commands.Bot):
    """Discord bot for handling democratic interactions and AI processing.

    This bot integrates with various AI models and processing pipelines to handle
    user interactions in a democratic context. It includes comprehensive resource
    management, error handling, and cleanup procedures.

    Key Features:
        - Resource monitoring and limits
        - Dependency-based extension loading
        - Message size validation
        - Memory usage tracking
        - Task timeout management
        - Rate limiting and spam protection
        - Proper resource cleanup

    Attributes:
        session: aiohttp ClientSession for making HTTP requests
        command_stats: Counter for tracking command usage
        socket_stats: Counter for tracking socket events
        graph: LangGraph for processing messages
        message_handler: Handler for processing messages
        attachment_handler: Handler for processing attachments
        version: Bot version
        guild_data: Guild configuration data
        bot_app_info: Discord application info
        owner_id: Bot owner's Discord ID
        invite: Bot invite link
        uptime: Bot start time
        resource_manager: Manager for system resources and limits
    """
    user: discord.ClientUser
    old_tree_error = Callable[[discord.Interaction, discord.app_commands.AppCommandError], Coroutine[Any, Any, None]]

    def __init__(
        self,
        command_prefix: str | None = None,
        description: str | None = None,
        intents: discord.Intents | None = None,
        *args: Any,
        **kwargs: Any
    ) -> None:
        """Initialize the DemocracyBot with required intents and models.

        Args:
            command_prefix: Optional custom command prefix. Defaults to aiosettings.prefix.
            description: Optional bot description. Defaults to DESCRIPTION.
            intents: Optional custom intents. Defaults to standard intents configuration.
            *args: Additional positional arguments passed to commands.Bot
            **kwargs: Additional keyword arguments passed to commands.Bot

        Raises:
            ValueError: If configuration values are invalid
            RuntimeError: If initialization fails
        """
        # Validate configuration
        if not aiosettings.prefix:
            raise ValueError("Bot prefix not configured")
        if not aiosettings.discord_client_id:
            raise ValueError("Discord client ID not configured")

        # Initialize resource manager with limits
        self.resource_manager = ResourceManager(
            ResourceLimits(
                max_memory_mb=getattr(aiosettings, "max_memory_mb", 512),
                max_tasks=getattr(aiosettings, "max_tasks", 100),
                max_response_size_mb=getattr(aiosettings, "max_response_size_mb", 8),
                max_buffer_size_kb=getattr(aiosettings, "max_buffer_size_kb", 64),
                task_timeout_seconds=getattr(aiosettings, "task_timeout_seconds", 300)
            )
        )

        allowed_mentions = AllowedMentions(roles=False, everyone=False, users=True)

        # Set up default intents if not provided
        if intents is None:
            intents = Intents.default()
            intents.message_content = True
            intents.guilds = True
            intents.members = True
            intents.bans = True
            intents.emojis = True
            intents.voice_states = True
            intents.messages = True
            intents.reactions = True

        self._command_prefix = command_prefix or aiosettings.prefix
        self._user = None

        super().__init__(
            command_prefix=self._command_prefix,
            description=description or DESCRIPTION,
            pm_help=None,
            help_attrs=dict(hidden=True),
            chunk_guilds_at_startup=False,
            heartbeat_timeout=150.0,
            allowed_mentions=allowed_mentions,
            intents=intents,
            enable_debug_events=True,
            *args,
            **kwargs
        )

        # Initialize session and stats
        self.command_stats: Counter = Counter()
        self.socket_stats: Counter = Counter()
        self.graph: CompiledStateGraph = memgraph

        # Initialize handlers with resource manager
        self.message_handler = MessageHandler(self)
        self.attachment_handler = AttachmentHandler()

        # Initialize bot attributes
        self.version: str = democracy_exe.__version__
        self.guild_data: dict[int, dict[str, Any]] = {}
        self.bot_app_info: AppInfo | None = None
        self.owner_id: int | None = None
        self.invite: str | None = None
        self.uptime: datetime.datetime | None = None
        self.pool: RedisConnectionPool | None = None

        self.resumes: defaultdict[int, list[datetime.datetime]] = defaultdict(list)
        self.identifies: defaultdict[int, list[datetime.datetime]] = defaultdict(list)

        # Configure rate limiting
        self.spam_control = commands.CooldownMapping.from_cooldown(
            rate=getattr(aiosettings, "rate_limit_rate", 10),
            per=getattr(aiosettings, "rate_limit_per", 12.0),
            type=commands.BucketType.user
        )

        # A counter to auto-ban frequent spammers
        # Triggering the rate limit 5 times in a row will auto-ban the user from the bot.
        self._auto_spam_count = Counter()
        self._spam_ban_threshold = getattr(aiosettings, "spam_ban_threshold", 5)

        self.channel_list = [int(x) for x in CHANNEL_ID.split(",")]
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=getattr(aiosettings, "max_queue_size", 1000))
        self.tasks: list[Any] = []
        self.num_workers = min(getattr(aiosettings, "num_workers", 3), 10)  # Max 10 workers

        self.total_sleep_time = 0

        self.start_time = datetime.datetime.now()
        self.typerCtx: dict | None = None
        self.job_queue: dict[Any, Any] = {}
        self.client_id: int | str = aiosettings.discord_client_id
        self.enable_ai = aiosettings.enable_ai

        # self.session = aiohttp.ClientSession()
        self.prefixes: list[str] = [aiosettings.prefix]
        self.version = democracy_exe.__version__
        self.max_retries: int = aiosettings.max_retries


    async def get_context(self, origin: discord.Interaction | Message, /, *, cls=Context) -> Context:
        """Retrieve the context for a Discord interaction or message.

        Args:
            origin: The Discord interaction or message to get the context from
            cls: The class type for the context object

        Returns:
            The context object retrieved for the provided origin

        Raises:
            RuntimeError: If message size exceeds limits
            ValueError: If message content is invalid
        """
        # Check memory usage before processing
        self.resource_manager.check_memory()

        # Add size checks for message content
        if isinstance(origin, Message):
            content_size = len(origin.content.encode('utf-8'))
            if aiosettings.enable_resource_management and content_size > self.resource_manager.limits.max_response_size_mb * 1024 * 1024:
                logger.error("Message size exceeds limit",
                           size=content_size,
                           limit=self.resource_manager.limits.max_response_size_mb * 1024 * 1024)
                raise RuntimeError(f"Message size {content_size} exceeds limit {self.resource_manager.limits.max_response_size_mb * 1024 * 1024}")

            # Validate message content
            if not origin.content or not origin.content.strip():
                raise ValueError("Empty message content")

            # Check attachments
            if origin.attachments:
                total_attachment_size = sum(a.size for a in origin.attachments)
                if aiosettings.enable_resource_management and total_attachment_size > self.resource_manager.limits.max_response_size_mb * 1024 * 1024:
                    logger.error("Total attachment size exceeds limit",
                               size=total_attachment_size,
                               limit=self.resource_manager.limits.max_response_size_mb * 1024 * 1024)
                    raise RuntimeError(f"Total attachment size {total_attachment_size} exceeds limit {self.resource_manager.limits.max_response_size_mb * 1024 * 1024}")

        ctx = await super().get_context(origin, cls=cls)
        ctx.prefix = self._command_prefix
        return ctx

    async def setup_hook(self) -> None:
        """Asynchronous setup hook for initializing the bot.

        This method is called to perform asynchronous setup tasks for the bot.
        It initializes the aiohttp session, sets up guild prefixes, retrieves
        bot application information, and loads extensions.

        Raises:
            RuntimeError: If initialization fails
            TimeoutError: If setup tasks timeout
            ValueError: If extension dependencies are not met
        """
        try:
            # Add timeout for setup tasks
            async with asyncio.timeout(30.0):
                # Initialize bot application info
                self.intents.members = True
                self.intents.message_content = True
                self.bot_app_info = await self.application_info()

                if hasattr(self.bot_app_info, "owner") and self.bot_app_info.owner:
                    self.owner_id = self.bot_app_info.owner.id


                # # self.owner_id = self.bot_app_info.owner.id

                # # Load extensions in dependency order
                # extension_order = get_extension_load_order(aiosettings.initial_extensions)
                # for extension in extension_order:
                #     await load_extension_with_retry(self, extension, self.max_retries)
                # Load extensions
                await self._load_extensions()

                # Initialize invite link
                app = await self.application_info()
                self.invite = discord.utils.oauth_url(
                    app.id,
                    permissions=discord.Permissions(administrator=True)
                )

        except TimeoutError:
            logger.error("Setup hook timed out")
            raise
        except Exception as e:
            logger.error("Setup hook failed", error=str(e))
            raise RuntimeError("Failed to initialize bot") from e

    async def add_task(self, coro: Coroutine) -> None:
        """Add a task to the bot's task list with proper tracking.

        Args:
            coro: The coroutine to create a task from

        Raises:
            RuntimeError: If max concurrent tasks limit is reached
        """
        # Track task in resource manager
        task_id = id(coro)
        await self.resource_manager.track_task(task_id)

        try:
            # Create and add new task with timeout
            async def wrapped_coro() -> None:
                try:
                    async with asyncio.timeout(self.resource_manager.limits.task_timeout_seconds):
                        await coro
                except TimeoutError:
                    logger.error("Task timed out",
                               task_id=task_id,
                               timeout=self.resource_manager.limits.task_timeout_seconds)
                    raise
                except Exception as e:
                    logger.error("Task failed",
                               task_id=task_id,
                               error=str(e))
                    raise
                finally:
                    # Release task resources
                    self.resource_manager.cleanup_tasks([task_id])
                    gc.collect()

            task = asyncio.create_task(wrapped_coro())

            # Add done callback to log completion
            def on_task_done(t: asyncio.Task) -> None:
                try:
                    exc = t.exception()
                    if exc:
                        logger.error("Task failed with exception",
                                   task_id=task_id,
                                   error=str(exc))
                except asyncio.CancelledError:
                    logger.info("Task was cancelled", task_id=task_id)

            task.add_done_callback(on_task_done)

        except Exception as e:
            # Clean up task tracking on error
            self.resource_manager.cleanup_tasks([task_id])
            raise

    async def cleanup(self) -> None:
        """Clean up bot resources before shutdown.

        This method ensures proper cleanup of all resources including tasks,
        connections, and handlers.

        Raises:
            TimeoutError: If cleanup tasks timeout
        """
        try:
            # Add timeout for cleanup
            async with asyncio.timeout(30.0):
                # Force cleanup of all resources
                await self.resource_manager.force_cleanup()

                # Close Redis pool if exists
                if self.pool:
                    try:
                        async with asyncio.timeout(5.0):
                            await self.pool.disconnect()
                    except TimeoutError:
                        logger.error("Redis pool disconnect timed out")
                    except Exception as e:
                        logger.error("Redis pool disconnect failed", error=str(e))

                # Clean up handlers in sequence
                for handler_name in ['message_handler', 'attachment_handler']:
                    handler = getattr(self, handler_name, None)
                    if handler and hasattr(handler, 'cleanup'):
                        try:
                            async with asyncio.timeout(5.0):
                                await handler.cleanup()
                        except TimeoutError:
                            logger.error(f"{handler_name} cleanup timed out")
                        except Exception as e:
                            logger.error(f"{handler_name} cleanup failed",
                                       error=str(e))

                logger.info("Cleanup completed successfully")

        except TimeoutError:
            logger.error("Cleanup timed out")
            raise
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))
            raise
        finally:
            # Force cleanup of any remaining resources
            gc.collect()

    async def on_command_error(self, ctx: commands.Context[commands.Bot], error: commands.CommandError) -> None:
        """Handle errors raised during command invocation.

        Args:
            ctx: The context in which the command was invoked
            error: The error that was raised during command invocation
        """
        if isinstance(error, commands.NoPrivateMessage):
            await ctx.author.send("This command cannot be used in private messages.")
        elif isinstance(error, commands.DisabledCommand):
            await ctx.author.send("Sorry. This command is disabled and cannot be used.")
        elif isinstance(error, commands.CommandInvokeError):
            if hasattr(error, "original"):
                original = error.original
                if not isinstance(original, discord.HTTPException):
                    logger.exception("In %s:", ctx.command.qualified_name, exc_info=original)
        elif isinstance(error, commands.ArgumentParsingError):
            await ctx.send(str(error))
        else:
            raise error

        # await logger.complete()


    def _clear_gateway_data(self) -> None:
        """
        Clear gateway data older than one week.

        This method removes entries from the `identifies` and `resumes` dictionaries
        that are older than one week. It iterates through each shard's list of dates
        and deletes the entries that are older than the specified time frame.

        Returns
        -------
            None

        """
        one_week_ago = discord.utils.utcnow() - datetime.timedelta(days=7)

        # Clear identifies data
        for shard_id, dates in list(self.identifies.items()):
            to_remove = [index for index, dt in enumerate(dates) if dt < one_week_ago]
            for index in reversed(to_remove):
                del dates[index]
            if not dates:
                del self.identifies[shard_id]

        # Clear resumes data
        for shard_id, dates in list(self.resumes.items()):
            to_remove = [index for index, dt in enumerate(dates) if dt < one_week_ago]
            for index in reversed(to_remove):
                del dates[index]
            if not dates:
                del self.resumes[shard_id]

        # Force garbage collection after clearing data
        gc.collect()

    async def before_identify_hook(self, shard_id: int, *, initial: bool) -> None:  # type: ignore
        """
        Perform actions before identifying the shard.

        This method is called before the bot identifies the shard with the Discord gateway.
        It clears old gateway data and appends the current timestamp to the identifies list
        for the given shard ID.

        Args:
        ----
            shard_id (int): The ID of the shard that is about to identify.
            initial (bool): Whether this is the initial identification of the shard.

        Returns:
        -------
            None

        """
        self._clear_gateway_data()
        self.identifies[shard_id].append(discord.utils.utcnow())
        await super().before_identify_hook(shard_id, initial=initial)


    def get_guild_prefixes(self, guild: discord.abc.Snowflake | None, *, local_inject=_prefix_callable) -> list[str]:
        """
        Retrieve the command prefixes for a specific guild.

        This function generates a list of command prefixes for the bot based on the provided guild.
        If the guild is None, it returns the default prefixes. The function uses a proxy message
        to simulate a message from the guild and retrieves the prefixes using the local_inject function.

        Args:
        ----
            guild (Optional[discord.abc.Snowflake]): The guild for which to retrieve the command prefixes.
            local_inject (Callable): A callable function to inject the local context for prefix retrieval.

        Returns:
        -------
            list[str]: A list of command prefixes for the specified guild.

        """
        proxy_msg = ProxyObject(guild)
        return local_inject(self, proxy_msg)  # type: ignore  # lying

    async def query_member_named(
        self, guild: discord.Guild, argument: str, *, cache: bool = False
    ) -> discord.Member | None:
        """
        Query a member by their name, name + discriminator, or nickname.

        This asynchronous function searches for a member in the specified guild
        by their name, name + discriminator (e.g., username#1234), or nickname.
        It can optionally cache the results of the query.

        Args:
        ----
            guild (discord.Guild): The guild to query the member in.
            argument (str): The name, nickname, or name + discriminator combo to check.
            cache (bool): Whether to cache the results of the query. Defaults to False.

        Returns:
        -------
            Optional[discord.Member]: The member matching the query or None if not found.

        """
        if len(argument) > 5 and argument[-5] == "#":
            username, _, discriminator = argument.rpartition("#")
            members = await guild.query_members(username, limit=100, cache=cache)
            return discord.utils.get(members, name=username, discriminator=discriminator)
        else:
            members = await guild.query_members(argument, limit=100, cache=cache)

            return discord.utils.find(lambda m: m.name == argument or m.nick == argument, members)  # pylint: disable=consider-using-in # pyright: ignore[reportAttributeAccessIssue]

    async def get_or_fetch_member(self, guild: discord.Guild, member_id: int) -> discord.Member | None:
        """
        Retrieve a member from the cache or fetch from the API if not found.

        This asynchronous function attempts to retrieve a member from the cache
        in the specified guild using the provided member ID. If the member is not
        found in the cache, it fetches the member from the Discord API. The function
        handles rate limiting and returns the member if found, or None if not found.

        Args:
        ----
            guild (discord.Guild): The guild to look in.
            member_id (int): The member ID to search for.

        Returns:
        -------
            Optional[discord.Member]: The member if found, or None if not found.

        """
        member = guild.get_member(member_id)
        if member is not None:
            return member

        shard: discord.ShardInfo = self.get_shard(guild.shard_id)  # type: ignore  # will never be None
        if shard.is_ws_ratelimited():
            try:
                member = await guild.fetch_member(member_id)
            except discord.HTTPException:
                return None
            else:
                return member

        members = await guild.query_members(limit=1, user_ids=[member_id], cache=True)
        return members[0] if members else None


    def get_session_id(self, message: discord.Message | discord.Thread) -> str:
        """
        Generate a session ID for the given message.

        This function generates a session ID based on the message context.
        The session ID is used as a key for the history session and as an identifier for logs.

        Args:
        ----
            message (discord.Message): The message or event dictionary.

        Returns:
        -------
            str: The generated session ID.

        Notes:
        -----
            - If the message is a direct message (DM), the session ID is based on the user ID.
            - If the message is from a guild (server) channel, the session ID is based on the channel ID.

        """
        # ctx: Context = await self.get_context(message)  # type: ignore
        if isinstance(message, discord.Thread):
            is_dm: bool = str(message.starter_message.channel.type) == "private"  # pyright: ignore[reportAttributeAccessIssue]
            user_id: int = message.starter_message.author.id  # pyright: ignore[reportAttributeAccessIssue]
            channel_id = message.starter_message.channel.name  # pyright: ignore[reportAttributeAccessIssue]
        elif isinstance(message, discord.Message):
            is_dm: bool = str(message.channel.type) == "private"  # pyright: ignore[reportAttributeAccessIssue]
            user_id: int = message.author.id  # pyright: ignore[reportAttributeAccessIssue]
            channel_id = message.channel.name  # pyright: ignore[reportAttributeAccessIssue]

        return f"discord_{user_id}" if is_dm else f"discord_{channel_id}"  # pyright: ignore[reportAttributeAccessIssue] # pylint: disable=possibly-used-before-assignment

    async def resolve_member_ids(
        self, guild: discord.Guild, member_ids: Iterable[int]
    ) -> AsyncIterator[discord.Member]:
        """
        Bulk resolve member IDs to member instances, if possible.

        This asynchronous function attempts to resolve a list of member IDs to their corresponding
        member instances within a specified guild. Members that cannot be resolved are discarded
        from the list. The function yields the resolved members lazily using an asynchronous iterator.

        Note:
        ----
            The order of the resolved members is not guaranteed to be the same as the input order.

        Args:
        ----
            guild (discord.Guild): The guild to resolve members from.
            member_ids (Iterable[int]): An iterable of member IDs to resolve.

        Yields:
        ------
            discord.Member: The resolved members.

        """
        needs_resolution = []
        for member_id in member_ids:
            member = guild.get_member(member_id)
            if member is not None:
                yield member
            else:
                needs_resolution.append(member_id)

        total_need_resolution = len(needs_resolution)
        if total_need_resolution == 1:
            shard: discord.ShardInfo = self.get_shard(guild.shard_id)  # type: ignore  # will never be None
            if shard.is_ws_ratelimited():
                try:
                    member = await guild.fetch_member(needs_resolution[0])
                except discord.HTTPException:
                    pass
                else:
                    yield member
            else:
                members = await guild.query_members(limit=1, user_ids=needs_resolution, cache=True)
                if members:
                    yield members[0]
        elif total_need_resolution <= 100:
            # Only a single resolution call needed here
            resolved = await guild.query_members(limit=100, user_ids=needs_resolution, cache=True)
            for member in resolved:
                yield member
        else:
            # We need to chunk these in bits of 100...
            for index in range(0, total_need_resolution, 100):
                to_resolve = needs_resolution[index : index + 100]
                members = await guild.query_members(limit=100, user_ids=to_resolve, cache=True)
                for member in members:
                    yield member

    async def on_ready(self) -> None:
        """Handle the event when the bot is ready."""
        if not self.user:
            logger.error("Bot user is not initialized")
            return

        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")
        self.invite = f"https://discordapp.com/api/oauth2/authorize?client_id={self.user.id}&scope=bot&permissions=0"
        self.guild_data = await preload_guild_data()
        print(
            f"""Logged in as {self.user}..
            Serving {len(self.users)} users in {len(self.guilds)} guilds
            Invite: {self.invite}
        """
        )
        game = Game("DemocracyExe")
        await self.change_presence(status=Status.online, activity=game)

        if not hasattr(self, "uptime"):
            self.uptime = discord.utils.utcnow()

        logger.info(f"Ready: {self.user} (ID: {self.user.id})")
        # await logger.complete()

    async def on_shard_resumed(self, shard_id: int) -> None:
        """
        Handle the event when a shard resumes.

        This method is called when a shard successfully resumes its connection
        to the Discord gateway. It logs the shard ID and the timestamp of the
        resume event.

        Args:
        ----
            shard_id (int): The ID of the shard that resumed.

        Returns:
        -------
            None

        """
        logger.info("Shard ID %s has resumed...", shard_id)
        self.resumes[shard_id].append(discord.utils.utcnow())
        # await logger.complete()

    @property
    def owner(self) -> discord.User:
        """
        Retrieve the owner of the bot.

        This property returns the owner of the bot as a discord.User object.
        The owner information is retrieved from the bot's application info.

        Returns
        -------
            discord.User: The owner of the bot.

        """
        return self.bot_app_info.owner  # pyright: ignore[reportAttributeAccessIssue]

    async def close(self) -> None:
        """Close the bot and its associated resources."""
        try:
            await super().close()
        finally:
            logger.info("Closing bot...")
            # if hasattr(self, "session") and not self.session.closed:
            #     await self.session.close()

    async def start(self, *args: Any, **kwargs: Any) -> None:
        """Start the bot and connect to Discord."""
        token = aiosettings.discord_token.get_secret_value() # pylint: disable=no-member
        await super().start(token, reconnect=True)

    async def _load_extensions(self) -> None:
        """Load bot extensions.

        This method loads all extensions from the cogs directory.
        It uses the extension_utils module to discover and load extensions.
        """
        from democracy_exe.chatbot.utils.discord_utils import extensions
        from democracy_exe.chatbot.utils.extension_manager import load_extensions

        logger.debug("Looking for extensions in cogs directory")
        extensions_found = list(extensions())
        logger.info(f"Found extensions: {extensions_found}")

        try:
            await load_extensions(self, extensions_found)
        except Exception as e:
            logger.error(f"Failed to load extensions: {e}")
            logger.exception("Extension loading failed")
            raise


    async def my_background_task(self) -> None:
        """Run a background task that sends a counter message to a specific channel every 60 seconds.

        This asynchronous method waits until the bot is ready, then continuously increments a counter
        and sends its value to a predefined Discord channel every 60 seconds.
        """
        await self.wait_until_ready()
        counter = 0

        channel_id = aiosettings.discord_general_channel
        channel = self.get_channel(channel_id)
        while not self.is_closed():
            counter += 1
            if channel and isinstance(channel, (TextChannel, DMChannel, Thread)):
                await channel.send(str(counter))
            await asyncio.sleep(60)  # task runs every 60 seconds

    async def on_worker_monitor(self) -> None:
        """Monitor and log the status of worker tasks.

        This asynchronous method waits until the bot is ready, then continuously
        logs the status of worker tasks every 10 seconds.
        """
        await self.wait_until_ready()
        counter = 0

        while not self.is_closed():
            counter += 1
            logger.info(f"Worker monitor iteration: {counter}")
            await asyncio.sleep(10)

    # # NOTE: attempting to refactor the on_message method to be more readable and maintainable
    # async def on_message(self, message: discord.Message) -> None:
    #     """Process incoming messages and route them through the AI pipeline."""
    #     if not self._should_process_message(message):
    #         return

    #     if message.reference is not None:
    #         await self._handle_reply(message, message.reference.resolved, message.content)
    #         return

    #     if self.enable_ai and self._is_gpt_channel(message.channel):
    #         await self._process_ai_message(message)
    #     else:
    #         logger.info("AI is disabled, skipping message processing... with llm")
    #         await self.process_commands(message)




    # # NOTE: enhanced version of the on_message method
    # # see: https://github.com/bossjones/goob_ai/blob/a63f43ab3592542097e762349d53955bcc97ad1d/src/goob_ai/goob_bot.py
    # # @pysnooper.snoop(thread_info=True, max_variable_length=None, depth=10)
    # # Process incoming messages and route them through the AI pipeline
    # async def on_message(self, message: discord.Message) -> None:
    #     """Process incoming messages and route them through the AI pipeline.

    #     Args:
    #         message: The Discord message to process
    #     """

    #     # Check if bot user is properly initialized
    #     if not self.user:
    #         logger.error("Bot user is not initialized")
    #         return

    #     # Ignore messages from the bot itself to prevent feedback loops
    #     if message.author == self.user:
    #         logger.info("Skipping message from bot itself")
    #         return

    #     # Ignore @everyone and @here
    #     if "@here" in message.content or "@everyone" in message.content:
    #         logger.info("Skipping message with @everyone or @here")
    #         return


    #     # TODO: This is where all the AI logic is going to go
    #     logger.info(f"Thread message to process - {message.author}: {message.content[:50]}")  # pyright: ignore[reportAttributeAccessIssue] # type: ignore
    #     if message.author.bot:
    #         logger.info(f"Skipping message from bot itself, message.author.bot = {message.author.bot}")
    #         return

    #     # skip messages that start w/ bot's prefix
    #     if message.content.startswith(aiosettings.prefix):  # pyright: ignore[reportAttributeAccessIssue]
    #         logger.info(f"Skipping message that starts with {aiosettings.prefix}")
    #         return


    #     # NOTE: on discord.message.reference, see: https://discordpy.readthedocs.io/en/stable/api.html#discord.Message.reference
    #     # #     reference: Optional[:class:`~discord.MessageReference`]
    #     # The message that this message references. This is only applicable to messages of
    #     # type :attr:`MessageType.pins_add`, crossposted messages created by a
    #     # followed channel integration, or message replies.

    #     # Check if this message is a reply to another message
    #     if message.reference is not None:
    #         # Get the message being replied to
    #         ref_message = message.reference.resolved

    #         # Check if the replied-to message was from this bot
    #         if ref_message.author.id == message.author.bot:
    #             # Remove any @ mentions of the bot from the message content
    #             content = message.content.replace(f"<@{message.author.bot}>", "").strip()
    #             # Update the message content with cleaned version
    #             message.content = content.strip()

    #             # Show typing indicator while generating response
    #             await message.channel.trigger_typing()
    #             # Generate AI response to the user's message
    #             response = await self.__generate_response(message)

    #             # Send the response as a reply to maintain thread context
    #             await message.reply(response)
    #         # Exit message handling since we've processed the reply
    #         return

    #     # if AI is enabled and message is in the GPT channel, process the message
    #     if self.enable_ai and (str(message.channel.type) == "text" and message.channel.id == 1240294186201124929):
    #         logger.info("AI is enabled, processing message...")
    #         # Check if the bot is mentioned in the message
    #         if self.user.mentioned_in(message):
    #             # Get or create a thread for this conversation
    #             thread = await self.message_handler._get_thread(message)
    #             if thread is None:
    #                 return

    #             # Extract thread and user IDs and convert to strings for consistency
    #             thread_id = thread.id
    #             user_id = message.author.id

    #             if isinstance(thread_id, int):
    #                 thread_id = str(thread_id)
    #             if isinstance(user_id, int):
    #                 user_id = str(user_id)

    #             # Format the input data for the AI processing pipeline
    #             input_data = {
    #                 "messages": [format_inbound_message(message)],
    #                 "configurable": {"thread_id": thread_id, "user_id": user_id}
    #             }

    #             # Process message through AI pipeline and handle any errors
    #             try:
    #                 response = await self.message_handler.stream_bot_response(self.graph, input_data)
    #             except Exception as e:
    #                 logger.exception(f"Error streaming bot response: {e}")
    #                 response = "An error occurred while processing your message."

    #             # Log that we're about to send the response
    #             logger.debug("Sending response to thread...")

    #             # Split response into chunks if it exceeds Discord's message length limit
    #             chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]

    #             # Send each chunk as a separate message in the thread
    #             for chunk in chunks:
    #                 await thread.send(chunk)
    #     else:
    #         logger.info("AI is disabled, skipping message processing... with llm")

    #         # This function processes the commands that have been registered to the bot and other groups. Without this coroutine, none of the commands will be triggered.
    #         await self.process_commands(message)

    # NOTE: original version of the on_message method
    # TODO: figure out how to refactor this to be more readable and maintainable and make tests pass
    # Process incoming messages and route them through the AI pipeline
    async def on_message(self, message: Message) -> None:
        """Process incoming messages and route them through the AI pipeline.

        Args:
            message: The Discord message to process
        """

        # Check if bot user is properly initialized
        if not self.user:
            logger.error("Bot user is not initialized")
            return

        # Ignore messages from the bot itself to prevent feedback loops
        if message.author == self.user:
            return


        if self.enable_ai:
            logger.info("AI is enabled, processing message...")
            # Check if the bot is mentioned in the message
            if self.user.mentioned_in(message):
                # Get or create a thread for this conversation
                thread = await self.message_handler.get_thread(message)
                if thread is None:
                    return

                # Extract thread and user IDs and convert to strings for consistency
                thread_id = thread.id
                user_id = message.author.id

                if isinstance(thread_id, int):
                    thread_id = str(thread_id)
                if isinstance(user_id, int):
                    user_id = str(user_id)

                # Format the input data for the AI processing pipeline
                input_data = {
                    "messages": [format_inbound_message(message)],
                    "configurable": {"thread_id": thread_id, "user_id": user_id}
                }

                # Process message through AI pipeline and handle any errors
                try:
                    response = await self.message_handler.stream_bot_response(self.graph, input_data)
                except Exception as e:
                    logger.exception(f"Error streaming bot response: {e}")
                    response = "An error occurred while processing your message."

                # Log that we're about to send the response
                logger.debug("Sending response to thread...")

                # Split response into chunks if it exceeds Discord's message length limit
                chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]

                # Send each chunk as a separate message in the thread
                for chunk in chunks:
                    await thread.send(chunk)
        else:
            logger.info("AI is disabled, skipping message processing... with llm")
            await self.process_commands(message)


    # NOTE: attempting to refactor the on_message method to be more readable and maintainable
    def _should_process_message(self, message: discord.Message) -> bool:
        """Determine if a Discord message should be processed by the bot's AI pipeline.

        This method implements filtering logic to determine whether a message should be
        processed by the bot's AI pipeline. It checks multiple conditions to prevent
        unwanted message processing and potential feedback loops.

        Args:
            message: The Discord message to evaluate. Contains metadata about the
                message including its author, content, and channel information.

        Returns:
            bool: True if the message should be processed by the bot's AI pipeline,
                False if the message should be ignored.

        Note:
            Messages are ignored if any of these conditions are met:
            - Bot user is not initialized
            - Message is from the bot itself
            - Message contains @here or @everyone mentions
            - Message is from another bot
            - Message starts with the bot's command prefix

        Example:
            ```python
            if self._should_process_message(message):
                await self._process_ai_message(message)
            ```
        """
        if not self.user:
            logger.error("Bot user is not initialized")
            return False

        if message.author == self.user:
            logger.info("Skipping message from bot itself")
            return False

        if "@here" in message.content or "@everyone" in message.content:
            logger.info("Skipping message with @everyone or @here")
            return False

        if message.author.bot:
            logger.info(f"Skipping message from bot itself, message.author.bot = {message.author.bot}")
            return False

        if message.content.startswith(aiosettings.prefix):
            logger.info(f"Skipping message that starts with {aiosettings.prefix}")
            return False

        return True

    async def _handle_reply(
        self,
        message: discord.Message,
        reply_to: discord.Message,
        content: str
    ) -> bool:
        """Handle a reply to a message.

        Processes a reply to a message, checking if it's a reply to the bot and
        handling the response appropriately.

        Args:
            message: The current message being processed
            reply_to: The message being replied to
            content: The content of the current message

        Returns:
            bool: True if the message was handled as a reply, False otherwise

        Raises:
            discord.DiscordException: If there's an error sending the response
        """
        # Skip if not replying to bot
        if reply_to.author.id != self.user.id:
            logger.info("Skipping reply to non-bot message")
            return False

        try:
            # Process reply and send response
            response = await self._process_message(message, content)
            await message.reply(response)
            return True
        except Exception as e:
            logger.exception(f"Error handling reply: {e!s}")
            raise

        return False

    def _is_gpt_channel(self, channel: discord.TextChannel) -> bool:
        """Check if the given channel is the designated GPT channel.

        This method determines if a channel is the designated GPT channel by checking
        its type and ID. The GPT channel is where AI-powered interactions are allowed
        to take place.

        Args:
            channel: The Discord text channel to evaluate. Must be a TextChannel
                instance containing channel metadata and properties.

        Returns:
            bool: True if the channel is the designated GPT channel, False otherwise.

        Note:
            A channel is considered a GPT channel if:
            - It is a text channel (channel.type == "text")
            - Its ID matches the predefined GPT channel ID (1240294186201124929)
        """
        return str(channel.type) == "text" and channel.id == 1240294186201124929

    async def _process_ai_message(self, message: discord.Message) -> None:
        """Process a message through the bot's AI pipeline.

        This method handles the AI processing workflow for a Discord message. It checks if the bot
        is mentioned, creates or retrieves a thread for the conversation, and processes the message
        through the AI pipeline to generate a response.

        Args:
            message: The Discord message to process. Must contain the message content,
                author information, and channel metadata.

        Returns:
            None

        Raises:
            Exception: If there's an error during the AI processing pipeline or response streaming.
                The error is logged and a generic error message is sent to the user.

        Note:
            The method performs the following steps:
            1. Checks if the bot is mentioned in the message
            2. Creates or retrieves a thread for the conversation
            3. Formats the message for AI processing
            4. Processes the message through the AI pipeline
            5. Splits and sends the response in chunks if necessary (Discord's 2000 char limit)
        """
        # Exit early if the bot isn't explicitly mentioned in the message
        if not self.user.mentioned_in(message):
            return

        # Get or create a thread for this conversation using the message handler
        thread = await self.message_handler.get_thread(message)
        # Exit if thread creation/retrieval failed
        if thread is None:
            return

        # Convert thread and user IDs to strings for consistent handling in the AI pipeline
        thread_id = str(thread.id)
        user_id = str(message.author.id)

        # Prepare the input data structure for the AI pipeline
        # format_inbound_message converts the Discord message to a format the AI can process
        input_data = {
            "messages": [format_inbound_message(message)],  # Convert message to AI-readable format
            "configurable": {"thread_id": thread_id, "user_id": user_id}  # Add metadata for context
        }

        try:
            # Process the message through the AI pipeline and get the response
            # Uses the LangGraph instance (self.graph) to generate the response
            response = await self.message_handler.stream_bot_response(self.graph, input_data)
        except Exception as e:
            # Log any errors that occur during processing and return a generic error message
            logger.exception(f"Error streaming bot response: {e}")
            response = "An error occurred while processing your message."

        # Log that we're about to send the response
        logger.debug("Sending response to thread...")

        # Split response into chunks of 2000 characters to comply with Discord's message length limit
        chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]

        # Send each chunk as a separate message in the thread
        for chunk in chunks:
            await thread.send(chunk)
