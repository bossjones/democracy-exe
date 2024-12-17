"""Core DemocracyBot implementation.

This module contains the main DemocracyBot class and its core functionality.
"""
from __future__ import annotations

import asyncio
import datetime

from collections import Counter
from typing import Any, Optional, cast

import aiohttp
import discord

from discord import Activity, AllowedMentions, AppInfo, DMChannel, Game, Guild, Intents, Message, Status, Thread, User
from discord.abc import Messageable
from discord.ext import commands
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from pydantic import SecretStr

import democracy_exe

from democracy_exe import utils
from democracy_exe.agentic.workflows.react.graph import graph as memgraph
from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.handlers.attachment_handler import AttachmentHandler
from democracy_exe.chatbot.handlers.message_handler import MessageHandler
from democracy_exe.chatbot.utils.guild_utils import preload_guild_data
from democracy_exe.utils.context import Context


DESCRIPTION = """An example bot to showcase the discord.ext.commands extension
module.

There are a number of utility commands being showcased here."""


class DemocracyBot(commands.Bot):
    """Discord bot for handling democratic interactions and AI processing.

    This bot integrates with various AI models and processing pipelines to handle
    user interactions in a democratic context.

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
    """

    def __init__(self) -> None:
        """Initialize the DemocracyBot with required intents and models."""
        allowed_mentions = AllowedMentions(roles=False, everyone=False, users=True)
        intents = Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        intents.bans = True
        intents.emojis = True
        intents.voice_states = True
        intents.messages = True
        intents.reactions = True
        super().__init__(
            command_prefix=aiosettings.prefix,
            description=DESCRIPTION,
            pm_help=None,
            help_attrs=dict(hidden=True),
            chunk_guilds_at_startup=False,
            heartbeat_timeout=150.0,
            allowed_mentions=allowed_mentions,
            intents=intents,
            enable_debug_events=True
        )

        # Initialize session and stats
        self.session: aiohttp.ClientSession = aiohttp.ClientSession()
        self.command_stats: Counter = Counter()
        self.socket_stats: Counter = Counter()
        self.graph: CompiledStateGraph = memgraph

        # Initialize handlers
        self.message_handler = MessageHandler(self)
        self.attachment_handler = AttachmentHandler()

        # Initialize bot attributes
        self.version: str = democracy_exe.__version__
        self.guild_data: dict[int, dict[str, Any]] = {}
        self.bot_app_info: AppInfo | None = None
        self.owner_id: int | None = None
        self.invite: str | None = None
        self.uptime: datetime.datetime | None = None

    async def get_context(self, origin: discord.Interaction | Message, /, *, cls=Context) -> Context:
        """Retrieve the context for a Discord interaction or message.

        Args:
            origin: The Discord interaction or message to get the context from
            cls: The class type for the context object

        Returns:
            The context object retrieved for the provided origin
        """
        return await super().get_context(origin, cls=cls)

    async def setup_hook(self) -> None:
        """Asynchronous setup hook for initializing the bot.

        This method is called to perform asynchronous setup tasks for the bot.
        It initializes the aiohttp session, sets up guild prefixes, retrieves
        bot application information, and loads extensions.
        """
        logger.debug("Starting setup_hook initialization")
        self.prefixes: list[str] = [aiosettings.prefix]

        self.version = democracy_exe.__version__
        self.guild_data = {}
        self.intents.members = True
        self.intents.message_content = True

        logger.debug("Retrieving bot application info")
        app_info = await self.application_info()
        self.bot_app_info = cast(AppInfo, app_info)
        if hasattr(self.bot_app_info, "owner") and self.bot_app_info.owner:
            self.owner_id = self.bot_app_info.owner.id

        # Load extensions will be moved to a separate utility function
        logger.info("Beginning extension loading process")
        await self._load_extensions()

        logger.info("Completed setup_hook initialization")
        await logger.complete()

    async def on_command_error(self, ctx: Context, error: commands.CommandError) -> None:
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

        await logger.complete()

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
        await logger.complete()

    async def close(self) -> None:
        """Close the bot and its associated resources."""
        await super().close()
        await self.session.close()

    async def start(self) -> None:
        """Start the bot and connect to Discord."""
        token = aiosettings.discord_token
        await super().start(str(token), reconnect=True)

    async def _load_extensions(self) -> None:
        """Load bot extensions.

        This method loads all extensions from the cogs directory.
        It uses the extension_utils module to discover and load extensions.
        """
        from democracy_exe.chatbot.utils.extension_utils import extensions, load_extensions

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

        channel = self.get_channel(aiosettings.discord_general_channel)
        while not self.is_closed():
            counter += 1
            if channel and isinstance(channel, Messageable):
                await channel.send(counter)
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

    async def on_message(self, message: Message) -> None:
        """Process incoming messages and route them through the AI pipeline.

        Args:
            message: The Discord message to process
        """
        if not self.user:
            logger.error("Bot user is not initialized")
            return

        if message.author == self.user:
            return

        if self.user.mentioned_in(message):
            thread = await self.message_handler._get_thread(message)
            if thread is None:
                return

            thread_id = thread.id
            user_id = message.author.id

            if isinstance(thread_id, int):
                thread_id = str(thread_id)
            if isinstance(user_id, int):
                user_id = str(user_id)

            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            user_input = {"messages": [self.message_handler._format_inbound_message(message)]}

            try:
                response = self.message_handler.stream_bot_response(self.graph, user_input, config)
            except Exception as e:
                logger.exception(f"Error streaming bot response: {e}")
                response = "An error occurred while processing your message."

            logger.debug("Sending response to thread...")
            # split messages into multiple outputs if len(output) is over discord's limit
            chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
            for chunk in chunks:
                await thread.send(chunk)
