# pylint: disable=no-value-for-parameter
# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false

"""democracy_exe.cli

This module implements the command-line interface for the democracy_exe application.
It provides commands for running bots, managing databases, and handling collections.

Implementation Details:
    - Command Structure: Uses AsyncTyperImproved for both sync and async commands
    - Bot Management: Supports both Discord and Terminal bot implementations
    - Error Handling: Comprehensive error capture with debug mode support
    - Resource Management: Proper cleanup and signal handling
    - Configuration: Environment-based settings with validation
    - Dependency Tracking: Version information and compatibility checks

Missing or Needs Improvement:
    - Comprehensive test coverage for all commands
    - Structured error handling framework
    - Command validation patterns
    - Resource cleanup patterns for long-running commands
    - Dependency validation and compatibility checks
    - Command retry and recovery mechanisms
    - Rate limiting for resource-intensive commands
    - Input validation framework
    - Command output formatting standardization
    - Progress tracking for long-running operations

Usage:
    Run commands using the CLI:
    $ democracyctl [command] [options]

    Example:
    $ democracyctl run-bot
    $ democracyctl version --verbose
"""

# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
# SOURCE: https://github.com/tiangolo/typer/issues/88#issuecomment-1732469681
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import traceback
import typing

from collections.abc import Awaitable, Callable, Iterable, Sequence
from enum import Enum
from functools import partial, wraps
from importlib import import_module, metadata
from importlib.metadata import version as importlib_metadata_version
from pathlib import Path
from re import Pattern
from typing import Annotated, Any, Dict, List, Optional, Set, Tuple, Type, Union
from urllib.parse import urlparse

import aiofiles
import anyio
import asyncer
import bpdb
import discord
import pysnooper
import rich
import structlog
import typer

from langchain.globals import set_debug, set_verbose
from langchain_chroma import Chroma as ChromaVectorStore


logger = structlog.get_logger(__name__)
from pinecone import ServerlessSpec
from pinecone.core.openapi.data.model.describe_index_stats_response import DescribeIndexStatsResponse
from pinecone.core.openapi.data.model.query_response import QueryResponse
from pinecone.core.openapi.data.model.upsert_response import UpsertResponse
from pinecone.data.index import Index
from redis.asyncio import ConnectionPool, Redis
from rich import print_json
from rich.console import Console
from rich.pretty import pprint
from rich.prompt import Prompt
from rich.table import Table
from typer import Typer

import democracy_exe

from democracy_exe.aio_settings import aiosettings, get_rich_console
from democracy_exe.asynctyper import AsyncTyper, AsyncTyperImproved
from democracy_exe.chatbot.discord_bot import DemocracyBot
from democracy_exe.chatbot.terminal_bot import ThreadSafeTerminalBot, go_terminal_bot
from democracy_exe.types import PathLike
from democracy_exe.utils import repo_typing
from democracy_exe.utils.base import print_line_seperator
from democracy_exe.utils.file_functions import fix_path


# from democracy_exe.utils.files_import import index_file_folder


# from democracy_exe.utils.collections_io import export_collection_data, import_collection_data
# SOURCE: https://python.langchain.com/v0.2/docs/how_to/debugging/
if aiosettings.debug_langchain:
    # Setting the global debug flag will cause all LangChain components with callback support (chains, models, agents, tools, retrievers) to print the inputs they receive and outputs they generate. This is the most verbose setting and will fully log raw inputs and outputs.
    set_debug(True)
    # Setting the verbose flag will print out inputs and outputs in a slightly more readable format and will skip logging certain raw outputs (like the token usage stats for an LLM call) so that you can focus on application logic.
    set_verbose(True)



class ChromaChoices(str, Enum):
    load = "load"
    generate = "generate"
    get_response = "get_response"


# Load existing subcommands
def load_commands(directory: str = "subcommands") -> None:
    """
    Load subcommands from the specified directory.

    This function loads subcommands from the specified directory and adds them to the main Typer app.
    It iterates over the files in the directory, imports the modules that end with "_cmd.py", and adds
    their Typer app to the main app if they have one.

    Args:
        directory (str, optional): The directory to load subcommands from. Defaults to "subcommands".

    Returns:
        None
    """
    script_dir = Path(__file__).parent
    subcommands_dir = script_dir / directory

    logger.debug(f"Loading subcommands from {subcommands_dir}")

    for filename in os.listdir(subcommands_dir):
        logger.debug(f"Filename: {filename}")
        if filename.endswith("_cmd.py"):
            module_name = f'{__name__.split(".")[0]}.{directory}.{filename[:-3]}'
            logger.debug(f"Loading subcommand: {module_name}")
            module = import_module(module_name)
            if hasattr(module, "APP"):
                logger.debug(f"Adding subcommand: {filename[:-7]}")
                APP.add_typer(module.APP, name=filename[:-7])


async def aload_commands(directory: str = "subcommands") -> None:
    """
    Asynchronously load subcommands from the specified directory.

    This function asynchronously loads subcommands from the specified directory and adds them to the main Typer app.
    It iterates over the files in the directory, imports the modules that end with "_cmd.py", and adds
    their Typer app to the main app if they have one.

    Args:
        directory (str, optional): The directory to load subcommands from. Defaults to "subcommands".

    Returns:
        None
    """
    script_dir = Path(__file__).parent
    subcommands_dir = script_dir / directory

    logger.debug(f"Loading subcommands from {subcommands_dir}")

    async with asyncer.create_task_group() as tg:
        for filename in os.listdir(subcommands_dir):
            logger.debug(f"Filename: {filename}")
            if filename.endswith("_cmd.py"):
                module_name = f'{__name__.split(".")[0]}.{directory}.{filename[:-3]}'
                logger.debug(f"Loading subcommand: {module_name}")

                async def _load_module(module_name: str, cmd_name: str) -> None:
                    module = import_module(module_name)
                    if hasattr(module, "APP"):
                        logger.debug(f"Adding subcommand: {cmd_name}")
                        APP.add_typer(module.APP, name=cmd_name)

                tg.start_soon(_load_module, module_name, filename[:-7])


APP = AsyncTyperImproved()
console = Console()
cprint = console.print
load_commands()


def version_callback(version: bool) -> None:
    """Print the version of democracy_exe."""
    if version:
        rich.print(f"democracy_exe version: {democracy_exe.__version__}")
        raise typer.Exit()


@APP.command()
def version(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show detailed version info")] = False,
) -> None:
    """Display version information."""
    rich.print(f"democracy_exe version: {democracy_exe.__version__}")
    if verbose:
        rich.print(f"Python version: {sys.version}")


@APP.command()
def deps() -> None:
    """Deps command"""
    rich.print(f"democracy_exe version: {democracy_exe.__version__}")
    rich.print(f"langchain_version: {importlib_metadata_version('langchain')}")
    rich.print(f"langchain_community_version: {importlib_metadata_version('langchain_community')}")
    rich.print(f"langchain_core_version: {importlib_metadata_version('langchain_core')}")
    rich.print(f"langchain_openai_version: {importlib_metadata_version('langchain_openai')}")
    rich.print(f"langchain_text_splitters_version: {importlib_metadata_version('langchain_text_splitters')}")
    rich.print(f"langchain_chroma_version: {importlib_metadata_version('langchain_chroma')}")
    rich.print(f"chromadb_version: {importlib_metadata_version('chromadb')}")
    rich.print(f"langsmith_version: {importlib_metadata_version('langsmith')}")
    rich.print(f"pydantic_version: {importlib_metadata_version('pydantic')}")
    rich.print(f"pydantic_settings_version: {importlib_metadata_version('pydantic_settings')}")
    rich.print(f"ruff_version: {importlib_metadata_version('ruff')}")


@APP.command()
def about() -> None:
    """About command"""
    typer.echo("This is GoobBot CLI")


@APP.command()
def show() -> None:
    """Show command"""
    cprint("\nShow democracy_exe", style="yellow")


# @pysnooper.snoop(thread_info=True, max_variable_length=None, watch=["APP"], depth=10)
def main():
    APP()
    load_commands()


# @pysnooper.snoop(thread_info=True, max_variable_length=None, depth=10)
def entry():
    """Required entry point to enable hydra to work as a console_script."""
    main()  # pylint: disable=no-value-for-parameter

@APP.command()
async def run_bot():
    """
    Run the Discord bot.

    This function starts the Discord bot and handles any exceptions that may occur during the bot's execution.
    It creates an instance of the SandboxAgent class and starts the bot using the start() method.

    If an exception occurs, it prints the exception details and enters the debugger if the dev_mode setting is enabled.

    Returns:
        None
    """

    logger.info("Running bot")
    try:
        async with DemocracyBot() as bot:
            # await bot.start(aiosettings.discord_token.get_secret_value())
            typer.echo("Running bot")
            await bot.start()
    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()

    # await logger.complete()


async def run_bot_with_redis():

    await asyncio.sleep(1)



@APP.command()
def run_terminal_bot() -> None:
    """Main entry point for terminal bot"""
    typer.echo("Starting up terminal bot")
    try:
        asyncio.run(go_terminal_bot())
    except Exception as ex:
        print(f"{ex}")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(f"Error Class: {ex.__class__}")
        output = f"[UNEXPECTED] {type(ex).__name__}: {ex}"
        print(output)
        print(f"exc_type: {exc_type}")
        print(f"exc_value: {exc_value}")
        traceback.print_tb(exc_traceback)
        if aiosettings.dev_mode:
            bpdb.pm()


@APP.command()
def run_load_commands() -> None:
    """Load subcommands"""
    typer.echo("Loading subcommands....")
    load_commands()


@APP.command()
def run_pyright() -> None:
    """Generate typestubs SandboxAgentAI"""
    typer.echo("Generating type stubs for SandboxAgentAI")
    repo_typing.run_pyright()


@APP.command()
def go() -> None:
    """Main entry point for DemocracyBot"""
    typer.echo("Starting up DemocracyBot")
    # NOTE: This is the approved way to run the bot, via asyncio.run.
    asyncio.run(run_bot())


@APP.command()
def db_current(verbose: Annotated[bool | None, typer.Option("--verbose/-v", help="Verbose mode")] = False) -> None:
    """Display current database revision.

    This command shows the current revision of the database.
    If the verbose option is enabled, additional details about the revision are displayed.

    Args:
        verbose (bool): If True, display additional details about the current revision.
    """
    typer.echo(f"Running db_current with verbose={verbose}")

    # current(verbose)


@APP.command()
def db_upgrade(revision: Annotated[str, typer.Option(help="Revision target")] = "head") -> None:
    """Upgrade to a later database revision.

    This command upgrades the database to the specified revision.
    By default, it upgrades to the latest revision ('head').

    Args:
        revision (str): The target revision to upgrade to. Defaults to 'head'.
    """
    typer.echo(f"Running db_upgrade with revision={revision}")

    # upgrade(revision)


@APP.command()
def db_downgrade(revision: Annotated[str, typer.Option(help="Revision target")] = "head") -> None:
    """Revert to a previous database revision.

    This command downgrades the database to the specified revision.

    Args:
        revision (str): The target revision to downgrade to.
    """
    typer.echo(f"Running db_downgrade with revision={revision}")

    # downgrade(revision)


@APP.command()
def export_collection(
    folder_path: Annotated[str, typer.Argument(help="Folder output path")],
    collection: Annotated[str, typer.Argument(help="Collection name")],
) -> None:
    """Export a collection to CSV postgres files.

    This command exports the specified collection to CSV files in the given folder path.

    Args:
        folder_path (str): The path to the folder where the CSV files will be saved.
        collection (str): The name of the collection to export.
    """
    typer.echo(f"Running export_collection with folder_path={folder_path}, collection={collection}")

    # export_collection_data(folder_path=folder_path, collection=collection)


@APP.command()
def import_collection(
    folder_path: Annotated[str, typer.Argument(help="Folder input path")],
    collection: Annotated[str, typer.Argument(help="Collection name")],
) -> None:
    """Import a collection from CSV postgres files.

    This command imports the specified collection from CSV files located in the given folder path.

    Args:
        folder_path (str): The path to the folder containing the CSV files.
        collection (str): The name of the collection to import.
    """
    typer.echo(f"Running import_collection with folder_path={folder_path}, collection={collection}")

    # import_collection_data(folder_path=folder_path, collection=collection)


@APP.command()
def import_file(
    file_path: Annotated[str, typer.Argument(help="File or folder path")],
    collection: Annotated[str, typer.Argument(help="Collection name")],
    options: Annotated[str | None, typer.Option("--options", "-o", help="Loader options in JSON format")] = None,
) -> None:
    """Add file or folder content to collection.

    This command adds the content of a file or folder to the specified collection.

    Args:
        file_path (str): The path to the file or folder.
        collection (str): The name of the collection to update.
        options (str): Loader options in JSON format.
    """
    typer.echo(f"Running import_file with file_path={file_path}, collection={collection}, options={options}")

    kwargs = {} if not options else json.loads(options)
    # num = index_file_folder(file_path=file_path, collection=collection, **kwargs)
    num = 0
    print(f"Collection '{collection}' updated from '{file_path}' with {num} documents.")


@APP.command()
def index(
    path: Annotated[list[str] | None, typer.Option()] = None,
    collection: Annotated[str, typer.Argument(help="Collection name")] = "default",
) -> None:
    """Add file or folder content to collection.

    This command adds the content of a file or folder to the specified collection.

    Args:
        file_path (str): The path to the file or folder.
        collection (str): The name of the collection to update.
        options (str): Loader options in JSON format.
    """
    typer.echo(f"Running index with path={path}, collection={collection}")

    # service = IndexWebService()
    # service.run(payload=path)


def handle_sigterm(signo, frame):
    sys.exit(128 + signo)  # this will raise SystemExit and cause atexit to be called


signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == "__main__":
    # import multiprocessing
    # from logging_tree import printout

    # # Determine best multiprocessing context based on platform
    # if sys.platform == "darwin":  # macOS
    #     mp_context = "spawn"  # Recommended for macOS
    # elif sys.platform == "win32":  # Windows
    #     mp_context = "spawn"  # Only option on Windows
    # else:  # Linux and other Unix
    #     mp_context = "fork"  # Default and most efficient on Unix

    # # Set up multiprocessing context
    # multiprocessing.set_start_method(mp_context, force=True)
    # context = multiprocessing.get_context(mp_context)

    # print(f"********************************************** Using multiprocessing context: {mp_context}")
    # print(f"********************************************** Using multiprocessing context: {context}")

    # # SOURCE: https://github.com/Delgan/loguru/blob/420704041797daf804b505e5220805528fe26408/docs/resources/recipes.rst#L1083
    # global_log_config(
    #     log_level=logging.getLevelName("DEBUG"),
    #     json=False,
    # )
    # from democracy_exe.bot_logger import global_log_config

    # # SOURCE: https://github.com/Delgan/loguru/blob/420704041797daf804b505e5220805528fe26408/docs/resources/recipes.rst#L1083
    # global_log_config(
    #     log_level=logging.getLevelName("DEBUG"),
    #     json=False,
    #     mp_context="spawn",
    # )
    from democracy_exe.bot_logger import logsetup

    APP()
