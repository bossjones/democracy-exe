# democracy_exe/chatbot/discord_bot.py
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations

import asyncio
import datetime
import json
import os
import pathlib
import re
import sys
import traceback
import uuid

from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple, TypeVar, Union, cast

import aiohttp
import bpdb
import discord
import rich

from codetiming import Timer
from discord.abc import Messageable
from discord.ext import commands
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS, PGVector
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler, StdOutCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from PIL import Image

from democracy_exe.ai.graphs import AgentState, router_graph
from democracy_exe.aio_settings import aiosettings


if TYPE_CHECKING:
    from discord import Message, TextChannel, User
    from discord.abc import Messageable

class DemocracyBot(commands.Bot):
    """Discord bot for handling democratic interactions and AI processing.

    This bot integrates with various AI models and processing pipelines to handle
    user interactions in a democratic context.

    Attributes:
        chat_model: OpenAI chat model instance
        embedding_model: OpenAI embeddings model instance
        vector_store: Vector store for embeddings
        session: aiohttp ClientSession for making HTTP requests
        command_stats: Counter for tracking command usage
        socket_stats: Counter for tracking socket events
    """

    def __init__(self):
        """Initialize the DemocracyBot with required intents and models."""
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix=aiosettings.discord_command_prefix, intents=intents)

        # Initialize AI models and stores
        self.chat_model = ChatOpenAI()
        self.embedding_model = OpenAIEmbeddings()
        self.vector_store = None

        # Initialize session and stats
        self.session = aiohttp.ClientSession()
        self.command_stats = Counter()
        self.socket_stats = Counter()

    async def setup_hook(self) -> None:
        """Set up the bot's initial configuration and load extensions.

        This method is called automatically when the bot starts up.
        """
        self.session = aiohttp.ClientSession()
        await logger.complete()

    async def close(self) -> None:
        """Clean up resources when shutting down the bot."""
        await super().close()
        await self.session.close()

    async def on_ready(self) -> None:
        """Handle bot ready event and set up initial state."""
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")

    async def _get_thread(self, message: discord.Message) -> discord.Thread:
        """Get or create a Discord thread for the given message.

        If the message is already in a thread, return that thread.
        Otherwise, create a new thread in the channel where the message was sent.

        Args:
            message (Message): The Discord message to get or create a thread for.

        Returns:
            discord.Thread: The thread associated with the message.
        """
        channel = message.channel # pyright: ignore[reportAttributeAccessIssue]
        if isinstance(channel, discord.Thread):
            return channel
        else:
            return await channel.create_thread(name="Response", message=message)


    # async def _create_or_fetch_lg_thread(self, thread_id: uuid.UUID) -> discord.Thread:
    #     """Create or fetch a LangGraph thread for the given thread ID.

    #     This function attempts to fetch an existing LangGraph thread. If it doesn't
    #     exist, a new thread is created.

    #     Args:
    #         thread_id (uuid.UUID): The unique identifier for the thread.

    #     Returns:
    #         Thread: The LangGraph thread object.
    #     """
    #     try:
    #         return await _LANGGRAPH_CLIENT.threads.get(thread_id)
    #     except Exception:
    #         pass
    #     return await _LANGGRAPH_CLIENT.threads.create(thread_id=thread_id)


    def _format_inbound_message(self, message: discord.Message) -> HumanMessage:
        """Format a Discord message into a HumanMessage for LangGraph processing.

        This function takes a Discord message and formats it into a structured
        HumanMessage object that includes context about the message's origin.

        Args:
            message (Message): The Discord message to format.

        Returns:
            HumanMessage: A formatted message ready for LangGraph processing.
        """
        guild_str = "" if message.guild is None else f"guild={message.guild}"  # pyright: ignore[reportAttributeAccessIssue]
        content = f"""<discord {guild_str} channel={message.channel} author={message.author!r}>
        {message.content}
        </discord>"""  # pyright: ignore[reportAttributeAccessIssue]
        return HumanMessage(
            content=content, name=str(message.author.global_name), id=str(message.id)
        )  # pyright: ignore[reportAttributeAccessIssue]

    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages and route them through the AI pipeline.

        Args:
            message: The Discord message to process
        """
        if message.author == self.user:
            return



        try:
            state = AgentState(
                query=message.content,
                response="",
                current_agent="",
                context={"message": message}
            )
            result = router_graph.process(state)
            await message.channel.send(result["response"])
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
            await message.channel.send("An error occurred while processing your message.")

    async def process_attachments(self, message: discord.Message) -> None:
        """Process any attachments in the message.

        Args:
            message: Discord message containing attachments
        """
        if not message.attachments:
            return

        try:
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    await self.process_image(attachment.url)
        except Exception as e:
            logger.exception(f"Error processing attachments: {e}")

    async def process_image(self, url: str) -> None:
        """Process an image from a given URL.

        Args:
            url: URL of the image to process

        Raises:
            aiohttp.ClientError: If there's an error downloading the image
            PIL.UnidentifiedImageError: If the image cannot be opened or processed
        """
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                image = Image.open(BytesIO(data)).convert("RGB")
                # Add image processing logic here
            else:
                logger.error(f"Failed to download image from {url}")

bot = DemocracyBot()
