# democracy_exe/chatbot/discord_bot.py
# pyright: reportAttributeAccessIssue=false
# pylint: disable=possibly-used-before-assignment
from __future__ import annotations

import asyncio
import base64
import datetime
import io
import json
import logging
import os
import pathlib
import re
import sys
import tempfile
import time
import traceback
import typing
import uuid

from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, NoReturn, Optional, Tuple, TypeVar, Union, cast

import aiofiles
import aiohttp
import bpdb
import discord
import pysnooper
import rich

from codetiming import Timer
from discord.abc import Messageable
from discord.ext import commands
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS, PGVector
from langchain_community.vectorstores import Redis as RedisVectorStore
from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler, StdOutCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from logging_tree import printout
from loguru import logger
from PIL import Image
from redis.asyncio import ConnectionPool as RedisConnectionPool
from rich.pretty import pprint
from starlette.responses import JSONResponse, StreamingResponse

import democracy_exe

from democracy_exe import shell, utils
from democracy_exe.agentic.workflows.react.graph import graph as memgraph
from democracy_exe.ai.graphs import AgentState, router_graph
from democracy_exe.aio_settings import aiosettings
from democracy_exe.bot_logger import REQUEST_ID_CONTEXTVAR, generate_tree, get_lm_from_tree, get_logger
from democracy_exe.clients.discord_client.utils import (
    close_thread,
    discord_message_to_message,
    is_last_message_stale,
    split_into_shorter_messages,
)
from democracy_exe.constants import (
    ACTIVATE_THREAD_PREFX,
    CHANNEL_ID,
    INPUT_CLASSIFICATION_NOT_A_QUESTION,
    INPUT_CLASSIFICATION_NOT_FOR_ME,
    MAX_THREAD_MESSAGES,
)
from democracy_exe.factories import guild_factory
from democracy_exe.models.loggers import LoggerModel
from democracy_exe.utils import async_, file_functions
from democracy_exe.utils.context import Context
from democracy_exe.utils.misc import CURRENTFUNCNAME


if TYPE_CHECKING:
    from discord import Message, TextChannel, User
    from discord.abc import Messageable



DESCRIPTION = """An example bot to showcase the discord.ext.commands extension
module.

There are a number of utility commands being showcased here."""

HERE = os.path.dirname(__file__)

INVITE_LINK = "https://discordapp.com/api/oauth2/authorize?client_id={}&scope=bot&permissions=0"


HOME_PATH = os.environ.get("HOME")

COMMAND_RUNNER = {"dl_thumb": shell.run_coroutine_subprocess}
THREAD_DATA = defaultdict()


def unlink_orig_file(a_filepath: str) -> str:
    """
    Delete the specified file and return its path.

    This function deletes the file at the given file path and returns the path of the deleted file.
    It uses the `os.unlink` method to remove the file and logs the deletion using `rich.print`.

    Args:
    ----
        a_filepath (str): The path to the file to be deleted.

    Returns:
    -------
        str: The path of the deleted file.

    """
    rich.print(f"deleting ... {a_filepath}")
    os.unlink(f"{a_filepath}")
    return a_filepath


# https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
def path_for(attm: discord.Attachment, basedir: str = "./") -> pathlib.Path:
    """
    Generate a pathlib.Path object for an attachment with a specified base directory.

    This function constructs a pathlib.Path object for a given attachment 'attm' using the specified base directory 'basedir'.
    It logs the generated path for debugging purposes and returns the pathlib.Path object.

    Args:
    ----
        attm (discord.Attachment): The attachment for which the path is generated.
        basedir (str): The base directory path where the attachment file will be located. Default is the current directory.

    Returns:
    -------
        pathlib.Path: A pathlib.Path object representing the path for the attachment file.

    Example:
    -------
        >>> attm = discord.Attachment(filename="example.png")
        >>> path = path_for(attm, basedir="/attachments")
        >>> print(path)
        /attachments/example.png

    """
    p = pathlib.Path(basedir, str(attm.filename))  # pyright: ignore[reportAttributeAccessIssue]
    logger.debug(f"path_for: p -> {p}")
    return p


# https://github.com/discord-math/bot/blob/babb41b71a68b4b099684b3e1ed583f84083f971/plugins/log.py#L63
async def save_attachment(attm: discord.Attachment, basedir: str = "./") -> None:
    """
    Save a Discord attachment to a specified directory.

    This asynchronous function saves a Discord attachment to the specified base directory.
    It constructs the path for the attachment, creates the necessary directories, and saves the attachment
    to the generated path. If an HTTPException occurs during saving, it retries the save operation.

    Args:
    ----
        attm (discord.Attachment): The attachment to be saved.
        basedir (str): The base directory path where the attachment file will be located. Default is the current directory.

    Returns:
    -------
        None

    """
    path = path_for(attm, basedir=basedir)
    logger.debug(f"save_attachment: path -> {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ret_code = await attm.save(path, use_cached=True)
        await asyncio.sleep(5)
    except discord.HTTPException:
        await attm.save(path)

    await logger.complete()


def attachment_to_dict(attm: discord.Attachment) -> dict[str, Any]:
    """
    Convert a discord.Attachment object to a dictionary.

    This function takes a discord.Attachment object and converts it into a dictionary
    containing relevant information about the attachment.

    The dictionary includes:
    - filename: The name of the file.
    - id: The unique identifier of the attachment.
    - proxy_url: The proxy URL of the attachment.
    - size: The size of the attachment in bytes.
    - url: The URL of the attachment.
    - spoiler: A boolean indicating if the attachment is a spoiler.
    - height: The height of the attachment (if applicable).
    - width: The width of the attachment (if applicable).
    - content_type: The MIME type of the attachment (if applicable).
    - attachment_obj: The original attachment object.

    Args:
    ----
        attm (discord.Attachment): The attachment object to be converted.

    Returns:
    -------
        Dict[str, Any]: A dictionary containing information about the attachment.

    """
    result = {
        "filename": attm.filename,  # pyright: ignore[reportAttributeAccessIssue]
        "id": attm.id,
        "proxy_url": attm.proxy_url,  # pyright: ignore[reportAttributeAccessIssue]
        "size": attm.size,  # pyright: ignore[reportAttributeAccessIssue]
        "url": attm.url,  # pyright: ignore[reportAttributeAccessIssue]
        "spoiler": attm.is_spoiler(),
    }
    if attm.height:  # pyright: ignore[reportAttributeAccessIssue]
        result["height"] = attm.height  # pyright: ignore[reportAttributeAccessIssue]
    if attm.width:  # pyright: ignore[reportAttributeAccessIssue]
        result["width"] = attm.width  # pyright: ignore[reportAttributeAccessIssue]
    if attm.content_type:  # pyright: ignore[reportAttributeAccessIssue]
        result["content_type"] = attm.content_type  # pyright: ignore[reportAttributeAccessIssue]

    result["attachment_obj"] = attm

    return result


def file_to_local_data_dict(fname: str, dir_root: str) -> dict[str, Any]:
    """
    Convert a file to a dictionary with metadata.

    This function takes a file path and a root directory, and converts the file
    into a dictionary containing metadata such as filename, size, extension, and
    a pathlib.Path object representing the file.

    Args:
    ----
        fname (str): The name of the file to be converted.
        dir_root (str): The root directory where the file is located.

    Returns:
    -------
        Dict[str, Any]: A dictionary containing metadata about the file.

    """
    file_api = pathlib.Path(fname)
    return {
        "filename": f"{dir_root}/{file_api.stem}{file_api.suffix}",
        "size": file_api.stat().st_size,
        "ext": f"{file_api.suffix}",
        "api": file_api,
    }


async def handle_save_attachment_locally(attm_data_dict: dict[str, Any], dir_root: str) -> str:
    """
    Save a Discord attachment locally.

    This asynchronous function saves a Discord attachment to a specified directory.
    It constructs the file path for the attachment, saves the attachment to the generated path,
    and returns the path of the saved file.

    Args:
    ----
        attm_data_dict (Dict[str, Any]): A dictionary containing information about the attachment.
        dir_root (str): The root directory where the attachment file will be saved.

    Returns:
    -------
        str: The path of the saved attachment file.

    """
    fname = f"{dir_root}/orig_{attm_data_dict['id']}_{attm_data_dict['filename']}"
    rich.print(f"Saving to ... {fname}")
    await attm_data_dict["attachment_obj"].save(fname, use_cached=True)
    await asyncio.sleep(1)
    return fname


# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/75c7ecd307f50390cfc798d39098fdb78535650c/cogs/AiCog.py#L237
async def download_image(url: str) -> BytesIO:  # type: ignore
    """
    Download an image from a given URL asynchronously.

    This asynchronous function uses aiohttp to make a GET request to the provided URL
    and downloads the image data. If the response status is 200 (OK), it reads the
    response data and returns it as a BytesIO object.

    Args:
    ----
        url (str): The URL of the image to download.

    Returns:
    -------
        BytesIO: A BytesIO object containing the downloaded image data.

    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.read()
                return io.BytesIO(data)


# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/7092ae6da21c86c7686549edd5c45335255b73ec/cogs/GlobalCog.py#L23
async def file_to_data_uri(file: discord.File) -> str:
    """
    Convert a discord.File object to a data URI.

    This asynchronous function reads the bytes from a discord.File object,
    base64 encodes the bytes, and constructs a data URI.

    Args:
    ----
        file (discord.File): The discord.File object to be converted.

    Returns:
    -------
        str: A data URI representing the file content.

    """
    with BytesIO(file.fp.read()) as f:  # pyright: ignore[reportAttributeAccessIssue]
        file_bytes = f.read()
    base64_encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:image;base64,{base64_encoded}"


# SOURCE: https://github.com/CrosswaveOmega/NikkiBot/blob/7092ae6da21c86c7686549edd5c45335255b73ec/cogs/GlobalCog.py#L23
async def data_uri_to_file(data_uri: str, filename: str) -> discord.File:
    """
    Convert a data URI to a discord.File object.

    This asynchronous function takes a data URI and a filename, decodes the base64 data,
    and creates a discord.File object with the decoded data.

    Args:
    ----
        data_uri (str): The data URI to be converted.
        filename (str): The name of the file to be created.

    Returns:
    -------
        discord.File: A discord.File object containing the decoded data.

    """
    # Split the data URI into its components
    metadata, base64_data = data_uri.split(",")
    # Get the content type from the metadata
    content_type = metadata.split(";")[0].split(":")[1]
    # Decode the base64 data
    file_bytes = base64.b64decode(base64_data)
    return discord.File(BytesIO(file_bytes), filename=filename, spoiler=False)


@async_.to_async
def get_logger_tree_printout() -> None:
    """
    Print the logger tree structure.

    This function prints the logger tree structure using the `printout` function
    from the `logging_tree` module. It is decorated with `@async_.to_async` to
    run asynchronously.
    """
    printout()


# SOURCE: https://realpython.com/how-to-make-a-discord-bot-python/#responding-to-messages
def dump_logger_tree() -> None:
    """
    Dump the logger tree structure.

    This function generates the logger tree structure using the `generate_tree` function
    and logs the tree structure using the `logger.debug` method.
    """
    rootm = generate_tree()
    logger.debug(rootm)


def dump_logger(logger_name: str) -> Any:
    """
    Dump the logger tree structure for a specific logger.

    This function generates the logger tree structure using the `generate_tree` function
    and retrieves the logger metadata for the specified logger name using the `get_lm_from_tree` function.
    It logs the retrieval process using the `logger.debug` method.

    Args:
    ----
        logger_name (str): The name of the logger to retrieve the tree structure for.

    Returns:
    -------
        Any: The logger metadata for the specified logger name.

    """
    logger.debug(f"getting logger {logger_name}")
    rootm: LoggerModel = generate_tree()
    return get_lm_from_tree(rootm, logger_name)


def filter_empty_string(a_list: list[str]) -> list[str]:
    """
    Filter out empty strings from a list of strings.

    This function takes a list of strings and returns a new list with all empty strings removed.

    Args:
    ----
        a_list (List[str]): The list of strings to be filtered.

    Returns:
    -------
        List[str]: A new list containing only non-empty strings from the input list.

    """
    filter_object = filter(lambda x: x != "", a_list)
    return list(filter_object)


# SOURCE: https://docs.python.org/3/library/asyncio-queue.html
async def worker(name: str, queue: asyncio.Queue) -> NoReturn:
    """
    Process tasks from the queue.

    This asynchronous function continuously processes tasks from the provided queue.
    Each task is executed using the COMMAND_RUNNER dictionary, and the function
    notifies the queue when a task is completed.

    Args:
    ----
        name (str): The name of the worker.
        queue (asyncio.Queue): The queue from which tasks are retrieved.

    Returns:
    -------
        NoReturn: This function runs indefinitely and does not return.

    """
    logger.info(f"starting working ... {name}")

    while True:
        # Get a "work item" out of the queue.
        co_cmd_task = await queue.get()
        print(f"co_cmd_task = {co_cmd_task}")

        # Sleep for the "co_cmd_task" seconds.

        await COMMAND_RUNNER[co_cmd_task.name](cmd=co_cmd_task.cmd, uri=co_cmd_task.uri)

        # Notify the queue that the "work item" has been processed.
        queue.task_done()

        print(f"{name} ran {co_cmd_task.name} with arguments {co_cmd_task}")


# SOURCE: https://realpython.com/python-async-features/#building-a-synchronous-web-server
async def co_task(name: str, queue: asyncio.Queue) -> AsyncIterator[None]:
    """
    Process tasks from the queue with timing.

    This asynchronous function processes tasks from the provided queue. It uses a timer to measure the elapsed time for each task. The function yields control back to the event loop after processing each task.

    Args:
    ----
        name (str): The name of the task.
        queue (asyncio.Queue): The queue from which tasks are retrieved.

    Yields:
    ------
        None: This function yields control back to the event loop after processing each task.

    """
    logger.info(f"starting working ... {name}")

    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
    while not queue.empty():
        co_cmd_task = await queue.get()
        print(f"Task {name} running")
        timer.start()
        await COMMAND_RUNNER[co_cmd_task.name](cmd=co_cmd_task.cmd, uri=co_cmd_task.uri)
        timer.stop()
        yield
        await logger.complete()


# SOURCE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py
async def get_prefix(_bot: DemocracyBot, message: discord.Message) -> Any:
    """
    Retrieve the command prefix for the bot based on the message context.

    This function determines the appropriate command prefix for the bot to use
    based on whether the message is from a direct message (DM) channel or a guild
    (server) channel. If the message is from a DM channel, it uses the default prefix
    from the settings. If the message is from a guild channel, it retrieves the prefix
    specific to that guild.

    Args:
    ----
        _bot (DemocracyBot): The instance of the bot.
        message (discord.Message): The message object from Discord.

    Returns:
    -------
        Any: The command prefix to be used for the bot.

    """
    logger.info(f"inside get_prefix(_bot, message) - > get_prefix({_bot}, {message})")
    logger.info(f"inside get_prefix(_bot, message) - > get_prefix({_bot}, {message})")
    logger.info(f"inside get_prefix(_bot, message) - > get_prefix({type(_bot)}, {type(message)})")
    prefix = (
        [aiosettings.prefix]
        if isinstance(message.channel, discord.DMChannel)  # pyright: ignore[reportAttributeAccessIssue]
        else [utils.get_guild_prefix(_bot, message.guild.id)]  # type: ignore
    )
    logger.info(f"prefix -> {prefix}")
    await logger.complete()
    return commands.when_mentioned_or(*prefix)(_bot, message)


# SOURCE: https://github.com/makupi/cookiecutter-discord.py-postgres/blob/master/%7B%7Bcookiecutter.bot_slug%7D%7D/bot/__init__.py#L28
async def preload_guild_data() -> dict[int, dict[str, str]]:
    """
    Preload guild data.

    This function initializes and returns a dictionary containing guild data.
    Each guild is represented by its ID and contains a dictionary with the guild's prefix.

    Returns
    -------
        Dict[int, Dict[str, str]]: A dictionary where the keys are guild IDs and the values are dictionaries
        containing guild-specific data, such as the prefix.

    """
    logger.info("preload_guild_data ... ")
    guilds = [guild_factory.Guild()]
    await logger.complete()
    return {guild.id: {"prefix": guild.prefix} for guild in guilds}


def extensions() -> Iterable[str]:
    """
    Yield extension module paths.

    This function searches for Python files in the 'cogs' directory relative to the current file's directory.
    It constructs the module path for each file and yields it.

    Yields
    ------
        str: The module path for each Python file in the 'cogs' directory.

    """
    module_dir = pathlib.Path(HERE)
    files = pathlib.Path(module_dir.stem, "cogs").rglob("*.py")
    for file in files:
        logger.debug(f"exension = {file.as_posix()[:-3].replace('/', '.')}")
        yield file.as_posix()[:-3].replace("/", ".")


def _prefix_callable(bot: DemocracyBot, msg: discord.Message) -> list[str]: # pyright: ignore[reportUninitializedInstanceVariable,reportUnusedFunction]
    """
    Generate a list of command prefixes for the bot.

    This function generates a list of command prefixes for the bot based on the message context.
    If the message is from a direct message (DM) channel, it includes the bot's user ID mentions
    and default prefixes. If the message is from a guild (server) channel, it includes the bot's
    user ID mentions and the guild-specific prefixes.

    Args:
    ----
        bot (DemocracyBot): The instance of the bot.
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


async def details_from_file(path_to_media_from_cli: str, cwd: str | None = None) -> tuple[str, str, str]:
    """
    Generate input and output file paths and retrieve the timestamp of the input file.

    This function takes a file path and an optional current working directory (cwd),
    and returns the input file path, output file path, and the timestamp of the input file.
    The timestamp is retrieved using platform-specific commands.

    Args:
    ----
        path_to_media_from_cli (str): The path to the media file provided via the command line.
        cwd (typing.Union[str, None], optional): The current working directory. Defaults to None.

    Returns:
    -------
        Tuple[str, str, str]: A tuple containing the input file path, output file path, and the timestamp of the input file.

    """
    p = pathlib.Path(path_to_media_from_cli)
    full_path_input_file = f"{p.stem}{p.suffix}"
    full_path_output_file = f"{p.stem}_smaller.mp4"
    rich.print(full_path_input_file)
    rich.print(full_path_output_file)
    if sys.platform == "darwin":
        get_timestamp = await shell._aio_run_process_and_communicate(
            ["gstat", "-c", "%y", f"{p.stem}{p.suffix}"], cwd=cwd
        )
    elif sys.platform == "linux":
        get_timestamp = await shell._aio_run_process_and_communicate(
            ["stat", "-c", "%y", f"{p.stem}{p.suffix}"], cwd=cwd
        )

    return full_path_input_file, full_path_output_file, get_timestamp # pylint: disable=possibly-used-before-assignment


class ProxyObject(discord.Object):
    def __init__(self, guild: discord.abc.Snowflake | None):
        super().__init__(id=0)
        self.guild: discord.abc.Snowflake | None = guild

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
        allowed_mentions = discord.AllowedMentions(roles=False, everyone=False, users=True)
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True
        intents.bans = True
        intents.emojis = True
        intents.voice_states = True
        intents.messages = True
        intents.reactions = True
        super().__init__(command_prefix=aiosettings.discord_command_prefix, description=DESCRIPTION,
            pm_help=None,
            help_attrs=dict(hidden=True),
            chunk_guilds_at_startup=False,
            heartbeat_timeout=150.0,
            allowed_mentions=allowed_mentions,
            intents=intents,
            enable_debug_events=True)

        # Initialize AI models and stores
        # self.chat_model = ChatOpenAI()
        # self.embedding_model = OpenAIEmbeddings()
        # self.vector_store = None

        # Initialize session and stats
        self.session = aiohttp.ClientSession()
        self.command_stats = Counter()
        self.socket_stats = Counter()
        self.graph: CompiledStateGraph = memgraph

    # async def setup_hook(self) -> None:
    #     """Set up the bot's initial configuration and load extensions.

    #     This method is called automatically when the bot starts up.
    #     """
    #     self.session = aiohttp.ClientSession()
    #     await logger.complete()


    async def get_context(self, origin: discord.Interaction | discord.Message, /, *, cls=Context) -> Context:
        """
        Retrieve the context for a Discord interaction or message.

        This asynchronous method retrieves the context for a Discord interaction or message
        and returns a Context object. It calls the superclass method to get the context
        based on the provided origin and class type.

        Args:
        ----
            origin (Union[discord.Interaction, discord.Message]): The Discord interaction or message to get the context from.
            cls (Context): The class type for the context object.

        Returns:
        -------
            Context: The context object retrieved for the provided origin.

        """
        return await super().get_context(origin, cls=cls)

    async def setup_hook(self) -> None:
        """
        Asynchronous setup hook for initializing the bot.

        This method is called to perform asynchronous setup tasks for the bot.
        It initializes the aiohttp session, sets up guild prefixes, retrieves
        bot application information, and loads extensions.

        It also sets the intents for members and message content to True.

        Raises
        ------
            Exception: If an extension fails to load, an exception is raised with
                        detailed error information.

        """

        self.prefixes: list[str] = [aiosettings.prefix]

        self.version = democracy_exe.__version__
        self.guild_data: dict[Any, Any] = {}
        self.intents.members = True
        self.intents.message_content = True

        self.bot_app_info: discord.AppInfo = await self.application_info()
        self.owner_id: int = self.bot_app_info.owner.id  # pyright: ignore[reportAttributeAccessIssue]

        for ext in extensions():
            try:
                await self.load_extension(ext)
            except Exception as ex:
                print(f"Failed to load extension {ext} - exception: {ex}")
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.error(f"Error Class: {ex.__class__!s}")
                output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                logger.warning(output)
                logger.error(f"exc_type: {exc_type}")
                logger.error(f"exc_value: {exc_value}")
                traceback.print_tb(exc_traceback)
                raise

        await logger.complete()


    async def on_command_error(self, ctx: Context, error: commands.CommandError) -> None:
        """
        Handle errors raised during command invocation.

        This method is called when an error is raised during the invocation of a command.
        It handles different types of command errors and sends appropriate messages to the user.

        Args:
        ----
            ctx (Context): The context in which the command was invoked.
            error (commands.CommandError): The error that was raised during command invocation.

        Returns:
        -------
            None

        """
        if isinstance(error, commands.NoPrivateMessage):
            await ctx.author.send("This command cannot be used in private messages.")
        elif isinstance(error, commands.DisabledCommand):
            await ctx.author.send("Sorry. This command is disabled and cannot be used.")
        elif isinstance(error, commands.CommandInvokeError):
            original = error.original  # pyright: ignore[reportAttributeAccessIssue]
            if not isinstance(original, discord.HTTPException):
                logger.exception("In %s:", ctx.command.qualified_name, exc_info=original)
        elif isinstance(error, commands.ArgumentParsingError):
            await ctx.send(str(error))

        await logger.complete()

    # async def close(self) -> None:
    #     """Clean up resources when shutting down the bot."""
    #     await super().close()
    #     await self.session.close()

    # async def on_ready(self) -> None:
    #     """Handle bot ready event and set up initial state."""
    #     logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
    #     print("------")

    async def on_ready(self) -> None:
        """
        Handle the event when the bot is ready.

        This method is called when the bot has successfully logged in and has completed
        its initial setup. It logs the bot's user information, sets the bot's presence,
        and prints the invite link. Additionally, it preloads guild data and logs the
        logger tree structure.

        Returns
        -------
            None

        """
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")
        self.invite = INVITE_LINK.format(self.user.id)
        self.guild_data = await preload_guild_data()
        rich.print(f"[bold green]self.guild_data:[/bold green] {self.guild_data}")
        print(
            f"""Logged in as {self.user}..
            Serving {len(self.users)} users in {len(self.guilds)} guilds
            Invite: {INVITE_LINK.format(self.user.id)}
        """
        )
        game = discord.Game("DemocracyExe") # pyright: ignore[reportAttributeAccessIssue]
        await self.change_presence(status=discord.Status.online, activity=game)

        # await bot.change_presence(activity=discord.Game(name="a game"))

        if not hasattr(self, "uptime"):
            self.uptime = discord.utils.utcnow()

        logger.info(f"Ready: {self.user} (ID: {self.user.id})")

        logger.info("LOGGING TREE:")
        await logger.complete()
        await get_logger_tree_printout()

    async def _get_thread(self, message: discord.Message) -> discord.Thread | discord.DMChannel | None:
        """Get or create a thread for the message.

        Args:
            message: The message to get/create a thread for

        Returns:
            Either a Thread object for server channels or DMChannel for direct messages
        """
        channel = message.channel

        # If this is a DM channel, just return it directly
        if isinstance(channel, discord.DMChannel):
            return channel

        # For regular channels, create a thread
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            return await channel.create_thread(name="Response", message=message)

        return None

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

        logger.debug(f"content: {content}")
        logger.complete()
        return HumanMessage(
            content=content, name=str(message.author.global_name), id=str(message.id)
        )  # pyright: ignore[reportAttributeAccessIssue]

    @pysnooper.snoop(max_variable_length=None)
    def stream_bot_response(
        self,
        graph: CompiledStateGraph = memgraph,
        user_input: dict[str, list[HumanMessage]] | None = None,
        thread: dict[str, Any] | None = None,
        interruptable: bool = False
    ) -> str:
        """Stream responses from the LangGraph Chatbot.

        Args:
            graph: The compiled state graph to use for generating responses.
                Defaults to memgraph.
            user_input: Dictionary containing user messages. Expected format is
                {"messages": [HumanMessage]}. Defaults to None.
            thread: Dictionary containing thread state information. Defaults to None.
            interruptable: Flag to indicate if streaming can be interrupted.
                Defaults to False.

        Returns:
            str: The concatenated response from all chunks generated by the graph.

        Example:
            >>> response = bot.stream_bot_response(
            ...     user_input={"messages": [HumanMessage(content="Hello")]}
            ... )
        """
        # Run the graph until the first interruption
        chunks = []
        for event in graph.stream(user_input, thread, stream_mode="values"):
            logger.debug(event)
            rich.print("[bold green]event:[/bold green]")
            pprint(event)
            # this is the response from the LLM
            chunk: AIMessage = event['messages'][-1]
            rich.print("[bold green]chunk:[/bold green]")
            rich.print(chunk)
            # pprint(chunk)
            chunk.pretty_print()
            if isinstance(chunk, AIMessage):
                # import bpdb; bpdb.set_trace()
                chunks.append(chunk.content)

        response = "".join(chunks)
        logger.error(f"response: {response}")
        return response

    async def on_message(self, message: discord.Message) -> None:
        """Process incoming messages and route them through the AI pipeline.

        Args:
            message: The Discord message to process
        """
        if message.author == self.user:
            return

        if self.user.mentioned_in(message):

            thread: discord.Thread | discord.DMChannel = await self._get_thread(message)
            thread_id = thread.id
            # thread_id = thread.id if isinstance(thread, discord.Thread) else None
            user_id = message.author.id  # TODO: is this unique?

            if isinstance(thread_id, int):
                thread_id = str(thread_id)
            if isinstance(user_id, int):
                user_id = str(user_id)

            config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
            user_input = {"messages": [self._format_inbound_message(message)]}
            # import bpdb; bpdb.set_trace()
            logger.error(f"user_input: {user_input}")
            logger.complete()

            try:
                response = self.stream_bot_response(self.graph, user_input, config)
            except Exception as e:
                logger.exception(f"Error streaming bot response: {e}")
                response = "An error occurred while processing your message."

            logger.debug("Sending response to thread...")
            # split messages into multiple outputs if len(output) is over discord's limit, i.e. 2000 characters
            chunks = [response[i:i+2000] for i in range(0, len(response), 2000)]
            for chunk in chunks:
                await thread.send(chunk)

    # SOURCE: https://github.com/aronweiler/assistant/blob/a8abd34c6973c21bc248f4782f1428a810daf899/src/discord/rag_bot.py#L90
    async def process_attachments(self, message: discord.Message) -> None:
        """
        Process attachments in a Discord message.

        This asynchronous function processes attachments in a Discord message by downloading each attached file,
        storing it in a temporary directory, and then loading and processing the files. It sends a message to indicate
        the start of processing and handles any errors that occur during the download process.

        Args:
        ----
            message (discord.Message): The Discord message containing attachments to be processed.

        Returns:
        -------
            None

        """
        if len(message.attachments) <= 0:  # pyright: ignore[reportAttributeAccessIssue]
            return

        root_temp_dir = f"temp/{uuid.uuid4()!s}"
        uploaded_file_paths = []
        for attachment in message.attachments:  # pyright: ignore[reportAttributeAccessIssue]
            logger.debug(f"Downloading file from {attachment.url}")
            # Download the file
            file_path = os.path.join(root_temp_dir, attachment.filename)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Download the file from the URL
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status != 200:
                        raise aiohttp.ClientException(f"Error downloading file from {attachment.url}")
                    data = await resp.read()

                    with open(file_path, "wb") as f:
                        f.write(data)

            uploaded_file_paths.append(file_path)

            # FIXME: RE ENABLE THIS SHIT 6/5/2024
            # # Process the files
            # await self.load_files(
            #     uploaded_file_paths=uploaded_file_paths,
            #     root_temp_dir=root_temp_dir,
            #     message=message,
            # )

    async def check_for_attachments(self, message: discord.Message) -> str:
        """
        Check a Discord message for attachments and process image URLs.

        This asynchronous function examines a Discord message for attachments,
        processes Tenor GIF URLs, downloads and processes image URLs, and modifies
        the message content based on the extracted information.

        Args:
        ----
            message (discord.Message): The Discord message to check for attachments and process.

        Returns:
        -------
            str: The updated message content with extracted information.

        """
        # Check if the message content is a URL

        message_content: str = message.content  # pyright: ignore[reportAttributeAccessIssue]
        url_pattern = re.compile(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
        if "https://tenor.com/view/" in message_content:

            # Extract the Tenor GIF URL from the message content
            start_index = message_content.index("https://tenor.com/view/")
            end_index = message_content.find(" ", start_index)
            if end_index == -1:
                tenor_url = message_content[start_index:]
            else:
                tenor_url = message_content[start_index:end_index]

            # Split the URL on forward slashes
            parts = tenor_url.split("/")

            # Extract the relevant words from the URL
            words = parts[-1].split("-")[:-1]

            # Join the words into a sentence
            sentence = " ".join(words)
            message_content = f"{message_content} [{message.author.display_name} posts an animated {sentence} ]"
            return message_content.replace(tenor_url, "")
        elif url_pattern.match(message_content):
            logger.info(f"Message content is a URL: {message_content}")

            # Download the image from the URL and convert it to a PIL image
            response = await download_image(message_content)

            # response = requests.get(message_content)
            image = Image.open(BytesIO(response.content)).convert("RGB")  # pyright: ignore[reportAttributeAccessIssue]
        else:
            logger.info(f"OTHERRRR Message content is a URL: {message_content}")
            # Download the image from the message and convert it to a PIL image
            image_url = message.attachments[0].url  # pyright: ignore[reportAttributeAccessIssue,reportInvalidTypeForm]

            response = await download_image(message_content)
            image = Image.open(BytesIO(response.content)).convert("RGB")  # pyright: ignore[reportAttributeAccessIssue]


        await logger.complete()
        return message_content

    def get_attachments(
        self, message: discord.Message
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str]]:
        """
        Retrieve attachment data from a Discord message.

        This function processes the attachments in a Discord message and converts each attachment
        to a dictionary format. It returns a tuple containing lists of dictionaries and file paths
        for further processing.

        Args:
        ----
            message (discord.Message): The Discord message containing attachments.

        Returns:
        -------
            Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]], List[str]]:
            A tuple containing:
                - A list of dictionaries with attachment data.
                - A list of local attachment file paths.
                - A list of dictionaries with local attachment data.
                - A list of media file paths.

        """
        attachment_data_list_dicts = []
        local_attachment_file_list = []
        local_attachment_data_list_dicts = []
        media_filepaths = []

        for attm in message.attachments:  # pyright: ignore[reportAttributeAccessIssue]
            data = attachment_to_dict(attm)
            attachment_data_list_dicts.append(data)

        return attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths

    async def write_attachments_to_disk(self, message: discord.Message) -> None:
        """
        Save attachments from a Discord message to disk.

        This asynchronous function processes the attachments in a Discord message,
        saves them to a temporary directory, and logs the file paths. It also handles
        any errors that occur during the download process.

        Args:
        ----
            message (discord.Message): The Discord message containing attachments to be saved.

        Returns:
        -------
            None

        """
        ctx = await self.get_context(message)
        attachment_data_list_dicts, local_attachment_file_list, local_attachment_data_list_dicts, media_filepaths = (
            self.get_attachments(message)
        )

        # Create a temporary directory to store the attachments
        tmpdirname = f"temp/{uuid.uuid4()!s}"
        os.makedirs(os.path.dirname(tmpdirname), exist_ok=True)
        print("created temporary directory", tmpdirname)
        with Timer(text="\nTotal elapsed time: {:.1f}"):
            # Save each attachment to the temporary directory
            for an_attachment_dict in attachment_data_list_dicts:
                local_attachment_path = await handle_save_attachment_locally(an_attachment_dict, tmpdirname)
                local_attachment_file_list.append(local_attachment_path)

            # Create a list of dictionaries with information about the local files
            for some_file in local_attachment_file_list:
                local_data_dict = file_to_local_data_dict(some_file, tmpdirname)
                local_attachment_data_list_dicts.append(local_data_dict)
                path_to_image = file_functions.fix_path(local_data_dict["filename"])
                media_filepaths.append(path_to_image)

            print("hello")

            rich.print("media_filepaths -> ")
            rich.print(media_filepaths)

            print("standy")

            try:
                for media_fpaths in media_filepaths:
                    # Compute all predictions first
                    full_path_input_file, full_path_output_file, get_timestamp = await details_from_file(
                        media_fpaths, cwd=f"{tmpdirname}"
                    )
            except Exception as ex:
                await ctx.send(embed=discord.Embed(description="Could not download story...."))
                print(ex)
                exc_type, exc_value, exc_traceback = sys.exc_info()
                logger.error(f"Error Class: {ex.__class__!s}")
                output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
                logger.warning(output)
                await ctx.send(embed=discord.Embed(description=output))
                logger.error(f"exc_type: {exc_type}")
                logger.error(f"exc_value: {exc_value}")
                traceback.print_tb(exc_traceback)

            # Log the directory tree of the temporary directory
            tree_list = file_functions.tree(pathlib.Path(f"{tmpdirname}"))
            rich.print("tree_list ->")
            rich.print(tree_list)

            file_to_upload_list = [f"{p}" for p in tree_list]
            logger.debug(f"{type(self).__name__} -> file_to_upload_list = {file_to_upload_list}")
            rich.print(file_to_upload_list)
            await logger.complete()

    def prepare_agent_input(
        self, message: discord.Message | discord.Thread, user_real_name: str, surface_info: dict
    ) -> dict[str, Any]:
        """
        Prepare the agent input from the incoming Discord message.

        This function constructs the input dictionary to be sent to the agent based on the
        provided Discord message, user's real name, and surface information. It includes
        the message content, user name, and any attachments if present.

        Args:
        ----
            message (discord.Message): The Discord message containing the user input.
            user_real_name (str): The real name of the user who sent the message.
            surface_info (Dict): The surface information related to the message.

        Returns:
        -------
            Dict[str, Any]: The input dictionary to be sent to the agent.

        """
        # ctx: Context = await self.get_context(message)  # type: ignore
        if isinstance(message, discord.Thread):
            agent_input = {"user name": user_real_name, "message": message.starter_message.content}  # pyright: ignore[reportAttributeAccessIssue]
            attachments: list[discord.Attachment] = message.starter_message.attachments  # pyright: ignore[reportAttributeAccessIssue]
        elif isinstance(message, discord.Message):
            agent_input = {"user name": user_real_name, "message": message.content}  # pyright: ignore[reportAttributeAccessIssue]
            attachments: list[discord.Attachment] = message.attachments  # pyright: ignore[reportAttributeAccessIssue]

        if len(attachments) > 0:  # pyright: ignore[reportAttributeAccessIssue]
            for attachment in attachments:  # pyright: ignore[reportAttributeAccessIssue]
                print(f"attachment -> {attachment}")  # pyright: ignore[reportAttributeAccessIssue]
                agent_input["file_name"] = attachment.filename  # pyright: ignore[reportAttributeAccessIssue]
                if attachment.content_type.startswith("image/"):  # pyright: ignore[reportAttributeAccessIssue]
                    agent_input["image_url"] = attachment.url  # pyright: ignore[reportAttributeAccessIssue]

        agent_input["surface_info"] = surface_info

        return agent_input

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
            channel_id = message.channel.id  # pyright: ignore[reportAttributeAccessIssue]

        return f"discord_{user_id}" if is_dm else f"discord_{channel_id}"  # pyright: ignore[reportAttributeAccessIssue]


    async def close(self) -> None:
        """
        Close the bot and its associated resources.

        This asynchronous method performs the necessary cleanup operations
        before shutting down the bot. It closes the aiohttp session and
        calls the superclass's close method to ensure proper shutdown.

        Returns
        -------
            None

        """
        await super().close()
        await self.session.close()

    async def start(self) -> None:  # type: ignore
        """
        Start the bot and connect to Discord.

        This asynchronous method initiates the bot's connection to Discord using the token
        specified in the settings. It ensures that the bot attempts to reconnect if the
        connection is lost.

        The method overrides the default `start` method from the `commands.Bot` class to
        include the bot's specific token and reconnection behavior.

        Returns
        -------
            None

        """
        await super().start(aiosettings.discord_token.get_secret_value(), reconnect=True) # pylint: disable=no-member

    async def my_background_task(self) -> None:
        """
        Run a background task that sends a counter message to a specific channel every 60 seconds.

        This asynchronous method waits until the bot is ready, then continuously increments a counter
        and sends its value to a predefined Discord channel every 60 seconds. The channel ID is retrieved
        from the bot's settings.

        The method ensures that the task runs indefinitely until the bot is closed.

        Args:
        ----
            None

        Returns:
        -------
            None

        """
        await self.wait_until_ready()
        # """
        # Wait until the bot is ready before starting the monitoring loop.
        # """
        counter = 0

        channel = self.get_channel(aiosettings.discord_general_channel)  # channel ID goes here
        while not self.is_closed():
            # """
            # Continuously monitor the worker tasks until the bot is closed.
            # """
            counter += 1
            # """
            # Increment the counter for each monitoring iteration.
            # """
            await channel.send(counter)  # pyright: ignore[reportAttributeAccessIssue]  # type: ignore
            await asyncio.sleep(60)  # task runs every 60 seconds

    async def on_worker_monitor(self) -> None:
        """
        Monitor and log the status of worker tasks.

        This asynchronous method waits until the bot is ready, then continuously
        increments a counter and logs the status of worker tasks every 10 seconds.
        It prints the list of tasks and the number of tasks currently being processed.

        The method ensures that the monitoring runs indefinitely until the bot is closed.

        Returns
        -------
            None

        """
        await self.wait_until_ready()
        counter = 0

        while not self.is_closed():
            counter += 1
            # await channel.send(counter)
            print(f" self.tasks = {self.tasks}")
            """
            Print the list of current tasks being processed.
            """
            print(f" len(self.tasks) = {len(self.tasks)}")
            """
            Print the number of tasks currently being processed.
            """
            await asyncio.sleep(10)
            """
            Sleep for 10 seconds before the next monitoring iteration.
            """



# SOURCE: https://github.com/darren-rose/DiscordDocChatBot/blob/63a2f25d2cb8aaace6c1a0af97d48f664588e94e/main.py#L28
# TODO: maybe enable this
async def send_long_message(channel: Any, message: discord.Message, max_length: int = 2000) -> None:
    """
    Send a long message by splitting it into chunks and sending each chunk.

    This asynchronous function takes a message and splits it into chunks of
    maximum length 'max_length'. It then sends each chunk as a separate message
    to the specified channel.

    Args:
    ----
        channel (Any): The channel to send the message chunks to.
        message (discord.Message): The message to be split into chunks and sent.
        max_length (int): The maximum length of each message chunk. Default is 2000.

    Returns:
    -------
        None

    """
    chunks = [message[i : i + max_length] for i in range(0, len(message), max_length)]  # type: ignore
    for chunk in chunks:
        await channel.send(chunk)

# bot = DemocracyBot()
