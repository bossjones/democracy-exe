"""Discord utility functions.

This module provides utility functions for Discord bot operations,
including message processing, permission checking, and file handling.
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import sys
import uuid

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Set, Tuple, Union, cast

import aiofiles
import aiohttp
import discord
import rich

from codetiming import Timer
from discord import Attachment, Client, File, Guild, Member, Message, PermissionOverwrite, Role, TextChannel, User
from discord.ext import commands
from logging_tree import printout
from loguru import logger

from democracy_exe.bot_logger import generate_tree, get_lm_from_tree
from democracy_exe.models.loggers import LoggerModel
from democracy_exe.utils import async_, shell


def extensions() -> list[str]:
    """Get list of extension paths.

    Returns:
        List of extension paths as strings
    """
    cogs_dir = Path(__file__).parent.parent / "cogs"
    return [
        f"democracy_exe.chatbot.cogs.{f.stem}"
        for f in cogs_dir.glob("*.py")
        if not f.name.startswith("_")
    ]

async def aio_extensions() -> list[str]:
    """Get list of extension paths asynchronously.

    Returns:
        List of extension paths as strings
    """
    return extensions()


def has_required_permissions(
    member: Member, channel: TextChannel, required_perms: set[str]
) -> bool:
    """Check if a member has the required permissions in a channel.

    Args:
        member: The member to check permissions for
        channel: The channel to check permissions in
        required_perms: Set of permission names to check

    Returns:
        bool: True if member has all required permissions, False otherwise

    Raises:
        ValueError: If invalid permission names are provided
    """
    try:
        channel_perms = channel.permissions_for(member)
        return all(
            getattr(channel_perms, perm, None) is True for perm in required_perms
        )
    except AttributeError as e:
        logger.error(f"Invalid permission name in {required_perms}: {e}")
        raise ValueError(f"Invalid permission name: {e}") from e
    except Exception as e:
        logger.error(f"Error checking permissions: {e}")
        raise


async def send_chunked_message(
    channel: TextChannel, content: str, chunk_size: int = 2000
) -> list[Message]:
    """Send a long message in chunks to avoid Discord's message length limit.

    Args:
        channel: The channel to send the message to
        content: The message content to send
        chunk_size: Maximum size of each message chunk (default: 2000)

    Returns:
        List of sent messages

    Raises:
        ValueError: If chunk_size is invalid
        discord.HTTPException: If message sending fails
    """
    try:
        if chunk_size <= 0 or chunk_size > 2000:
            raise ValueError("chunk_size must be between 1 and 2000")

        messages: list[Message] = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            try:
                msg = await channel.send(chunk)
                messages.append(msg)
                await asyncio.sleep(0.5)  # Rate limiting prevention
            except discord.HTTPException as e:
                logger.error(f"Failed to send message chunk: {e}")
                raise

        return messages
    except Exception as e:
        logger.error(f"Error in send_chunked_message: {e}")
        raise


def create_embed(
    title: str,
    description: str,
    color: discord.Color | None = None,
    fields: list[dict[str, str]] | None = None,
    footer: str | None = None,
    thumbnail_url: str | None = None,
) -> discord.Embed:
    """Create a Discord embed with the specified parameters.

    Args:
        title: The embed title
        description: The embed description
        color: The embed color (default: None)
        fields: List of field dictionaries with 'name' and 'value' keys (default: None)
        footer: Footer text (default: None)
        thumbnail_url: URL for thumbnail image (default: None)

    Returns:
        discord.Embed: The created embed

    Raises:
        ValueError: If required parameters are missing or invalid
    """
    try:
        embed = discord.Embed(
            title=title,
            description=description,
            color=color or discord.Color.blue()
        )

        if fields:
            for field in fields:
                if "name" not in field or "value" not in field:
                    raise ValueError("Field must contain 'name' and 'value' keys")
                embed.add_field(
                    name=field["name"],
                    value=field["value"],
                    inline=field.get("inline", True)
                )

        if footer:
            embed.set_footer(text=footer)

        if thumbnail_url:
            embed.set_thumbnail(url=thumbnail_url)

        return embed
    except Exception as e:
        logger.error(f"Error creating embed: {e}")
        raise


async def get_or_create_role(
    guild: Guild, role_name: str, **role_params: Any
) -> Role:
    """Get an existing role or create a new one if it doesn't exist.

    Args:
        guild: The guild to get/create the role in
        role_name: Name of the role
        **role_params: Additional role parameters (color, permissions, etc.)

    Returns:
        The found or created role

    Raises:
        discord.Forbidden: If bot lacks permission to manage roles
        discord.HTTPException: If role creation fails
    """
    try:
        existing_role = discord.utils.get(guild.roles, name=role_name)
        if existing_role:
            return existing_role

        logger.info(f"Creating new role: {role_name}")
        return await guild.create_role(name=role_name, **role_params)
    except discord.Forbidden as e:
        logger.error(f"Insufficient permissions to manage roles: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in get_or_create_role: {e}")
        raise


async def safe_delete_messages(
    messages: list[Message], delay: float | None = None
) -> None:
    """Safely delete messages with error handling and optional delay.

    Args:
        messages: List of messages to delete
        delay: Optional delay before deletion in seconds

    Raises:
        discord.Forbidden: If bot lacks permission to delete messages
        discord.HTTPException: If message deletion fails
    """
    try:
        if delay:
            await asyncio.sleep(delay)

        for msg in messages:
            try:
                await msg.delete()
                await asyncio.sleep(0.5)  # Rate limiting prevention
            except discord.NotFound:
                logger.debug(f"Message {msg.id} already deleted")
            except discord.Forbidden as e:
                logger.error(f"No permission to delete message {msg.id}: {e}")
                raise
            except Exception as e:
                logger.error(f"Error deleting message {msg.id}: {e}")
                continue
    except Exception as e:
        logger.error(f"Error in safe_delete_messages: {e}")
        raise


def get_member_roles_hierarchy(member: Member) -> list[Role]:
    """Get member's roles sorted by hierarchy position.

    Args:
        member: The member to get roles for

    Returns:
        List of roles sorted by position (highest first)

    Raises:
        ValueError: If member has no roles
    """
    try:
        roles = sorted(member.roles, key=lambda r: r.position, reverse=True)
        if not roles:
            raise ValueError(f"Member {member.name} has no roles")
        return roles
    except Exception as e:
        logger.error(f"Error getting member roles hierarchy: {e}")
        raise


async def setup_channel_permissions(
    channel: TextChannel,
    role_overwrites: dict[Role, PermissionOverwrite],
) -> None:
    """Set up channel permission overwrites for roles.

    Args:
        channel: The channel to set permissions for
        role_overwrites: Dictionary mapping roles to their permission overwrites

    Raises:
        discord.Forbidden: If bot lacks permission to manage channel permissions
        discord.HTTPException: If setting permissions fails
    """
    try:
        await channel.edit(overwrites=role_overwrites)
    except discord.Forbidden as e:
        logger.error(f"No permission to edit channel overwrites: {e}")
        raise
    except Exception as e:
        logger.error(f"Error setting channel permissions: {e}")
        raise


def format_user_info(user: User | Member) -> str:
    """Format user information into a readable string.

    Args:
        user: The user or member to format information for

    Returns:
        Formatted string containing user information

    Raises:
        ValueError: If user object is invalid
    """
    try:
        info = [
            f"Username: {user.name}",
            f"ID: {user.id}",
            f"Created: {user.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if isinstance(user, Member):
            info.extend([
                f"Nickname: {user.nick or 'None'}",
                f"Joined: {user.joined_at.strftime('%Y-%m-%d %H:%M:%S') if user.joined_at else 'Unknown'}",
                f"Top Role: {user.top_role.name}",
            ])

        return "\n".join(info)
    except Exception as e:
        logger.error(f"Error formatting user info: {e}")
        raise ValueError(f"Invalid user object: {e}")


def unlink_orig_file(a_filepath: str) -> str:
    """Delete the specified file and return its path.

    Args:
        a_filepath: The path to the file to be deleted

    Returns:
        The path of the deleted file

    Raises:
        OSError: If file deletion fails
    """
    try:
        rich.print(f"deleting ... {a_filepath}")
        os.unlink(f"{a_filepath}")
        return a_filepath
    except OSError as e:
        logger.error(f"Error deleting file {a_filepath}: {e}")
        raise


async def details_from_file(
    path_to_media_from_cli: str, cwd: str | None = None
) -> tuple[str, str, str]:
    """Generate input and output file paths and retrieve the timestamp of the input file.

    Args:
        path_to_media_from_cli: The path to the media file provided via command line
        cwd: The current working directory (default: None)

    Returns:
        Tuple containing:
            - Input file path
            - Output file path
            - File timestamp

    Raises:
        FileNotFoundError: If input file doesn't exist
        OSError: If file stats cannot be retrieved
    """
    try:
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
        else:
            raise OSError(f"Unsupported platform: {sys.platform}")

        return full_path_input_file, full_path_output_file, get_timestamp
    except Exception as e:
        logger.error(f"Error getting file details: {e}")
        raise


def filter_empty_string(a_list: list[str]) -> list[str]:
    """Filter out empty strings from a list of strings.

    Args:
        a_list: The list of strings to be filtered

    Returns:
        A new list containing only non-empty strings
    """
    return list(filter(lambda x: x != "", a_list))


async def worker(name: str, queue: asyncio.Queue) -> NoReturn:
    """Process tasks from the queue.

    Args:
        name: The name of the worker
        queue: The queue from which tasks are retrieved

    Raises:
        Exception: If task execution fails
    """
    logger.info(f"starting worker ... {name}")

    while True:
        try:
            co_cmd_task = await queue.get()
            logger.debug(f"co_cmd_task = {co_cmd_task}")

            await shell.run_coroutine_subprocess(cmd=co_cmd_task.cmd, uri=co_cmd_task.uri)
            queue.task_done()
            logger.info(f"{name} ran {co_cmd_task.name} with arguments {co_cmd_task}")
        except Exception as e:
            logger.error(f"Error in worker {name}: {e}")
            queue.task_done()
            continue


async def co_task(name: str, queue: asyncio.Queue) -> AsyncIterator[None]:
    """Process tasks from the queue with timing.

    Args:
        name: The name of the task
        queue: The queue from which tasks are retrieved

    Yields:
        None after each task is processed

    Raises:
        Exception: If task execution fails
    """
    logger.info(f"starting task ... {name}")

    timer = Timer(text=f"Task {name} elapsed time: {{:.1f}}")
    while not queue.empty():
        try:
            co_cmd_task = await queue.get()
            logger.info(f"Task {name} running")
            timer.start()
            await shell.run_coroutine_subprocess(cmd=co_cmd_task.cmd, uri=co_cmd_task.uri)
            timer.stop()
            yield
            await logger.complete()
        except Exception as e:
            logger.error(f"Error in task {name}: {e}")
            continue


@async_.to_async
def get_logger_tree_printout() -> None:
    """Print the logger tree structure."""
    printout()


def dump_logger_tree() -> None:
    """Dump the logger tree structure."""
    rootm = generate_tree()
    logger.debug(rootm)


def dump_logger(logger_name: str) -> Any:
    """Dump the logger tree structure for a specific logger.

    Args:
        logger_name: The name of the logger to retrieve the tree structure for

    Returns:
        The logger metadata for the specified logger name

    Raises:
        KeyError: If logger name is not found
    """
    try:
        logger.debug(f"getting logger {logger_name}")
        rootm: LoggerModel = generate_tree()
        return get_lm_from_tree(rootm, logger_name)
    except Exception as e:
        logger.error(f"Error dumping logger {logger_name}: {e}")
        raise
