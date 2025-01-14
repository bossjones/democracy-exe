# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Message handler for processing Discord messages."""
from __future__ import annotations

import asyncio
import io
import re

from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import discord
import structlog

from PIL import Image

from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.handlers.attachment_handler import AttachmentHandler
from democracy_exe.chatbot.utils.resource_manager import ResourceLimits, ResourceManager
from democracy_exe.constants import MAX_BYTES_UPLOAD_DISCORD, MAX_FILE_UPLOAD_IMAGES_IMGUR


logger = structlog.get_logger(__name__)

class MessageHandler:
    """Handles processing of Discord messages."""

    def __init__(self, bot: Any) -> None:
        """Initialize the message handler.

        Args:
            bot: The Discord bot instance
        """
        self.bot = bot
        self.attachment_handler = AttachmentHandler()
        self._download_semaphore = asyncio.Semaphore(
            getattr(aiosettings, "max_concurrent_downloads", 5)
        )
        limits = ResourceLimits(
            max_memory_mb=getattr(aiosettings, "max_memory_mb", 512),
            max_tasks=getattr(aiosettings, "max_tasks", 100),
            max_response_size_mb=getattr(aiosettings, "max_response_size_mb", 1),
            max_buffer_size_kb=getattr(aiosettings, "max_buffer_size_kb", 64),
            task_timeout_seconds=getattr(aiosettings, "task_timeout_seconds", 30.0)
        )
        self._resource_manager = ResourceManager(limits)
        self._max_total_size = MAX_BYTES_UPLOAD_DISCORD
        self._max_image_size = MAX_FILE_UPLOAD_IMAGES_IMGUR

    async def check_for_attachments(self, message: discord.Message) -> str:
        """Check message for attachments and process them.

        Args:
            message: Discord message to check

        Returns:
            str: Message content or processed image URL

        Raises:
            RuntimeError: If attachment size exceeds limit
        """
        task = asyncio.current_task()
        if task:
            self._resource_manager.track_task(task)

        try:
            attachments = message.attachments
            content = message.content
            total_size = sum(a.size for a in attachments)

            if total_size > self._max_total_size:
                raise RuntimeError(f"Total attachment size {total_size} exceeds {self._max_total_size} limit")

            # Check for image URLs in content
            image_pattern = r'https?://[^\s<\"]+?\.(?:png|jpg|jpeg|gif|webp)'
            image_urls = re.findall(image_pattern, content)

            if image_urls:
                return await self.handle_url_image(image_urls[0])

            if attachments:
                return await self.handle_attachment_image(attachments[0])

            return content

        finally:
            if task:
                await self._resource_manager.cleanup_tasks([task])
                logger.info("Resource cleanup completed", task=str(task))

    async def stream_bot_response(
        self,
        graph: Any,
        input_data: dict[str, Any]
    ) -> str:
        """Stream bot response from LangGraph.

        Args:
            graph: LangGraph instance
            input_data: Input data for the graph

        Returns:
            str: Final response

        Raises:
            RuntimeError: If response exceeds size limit or times out
        """
        task = asyncio.current_task()
        if task:
            self._resource_manager.track_task(task)

        try:
            timeout = getattr(aiosettings, "task_timeout_seconds", 30.0)
            max_size = getattr(aiosettings, "max_response_size_mb", 2) * 1024 * 1024

            response = await graph.ainvoke(input_data) if hasattr(graph, 'ainvoke') else graph.invoke(input_data)

            if not response or 'messages' not in response:
                raise RuntimeError("No response generated")

            messages = response['messages']
            combined_content = ''.join(str(m) for m in messages)
            response_size = len(combined_content.encode('utf-8'))

            if response_size > max_size:
                raise RuntimeError("Response exceeds Discord message size limit")

            return combined_content

        except TimeoutError:
            raise RuntimeError("Response generation timed out")

        except Exception as e:
            logger.exception("Error generating response", error=str(e))
            raise

        finally:
            if task:
                await self._resource_manager.cleanup_tasks([task])
                logger.info("Resource cleanup completed", task=str(task))

    async def handle_url_image(self, url: str) -> str:
        """Handle image from URL.

        Args:
            url: Image URL

        Returns:
            str: Processed image URL

        Raises:
            RuntimeError: If image size exceeds limit
        """
        async with self._download_semaphore:
            response = await self.attachment_handler.download_image(url)
            if response:
                size = len(response.getvalue())
                if size > self._max_image_size:
                    raise RuntimeError(f"Image size {size} exceeds {self._max_image_size} limit")
                return url

    async def handle_attachment_image(self, attachment: discord.Attachment) -> str:
        """Handle Discord attachment image.

        Args:
            attachment: Discord attachment

        Returns:
            str: Processed image URL

        Raises:
            RuntimeError: If image size exceeds limit
        """
        if attachment.size > self._max_image_size:
            raise RuntimeError(f"Image size {attachment.size} exceeds {self._max_image_size} limit")

        async with self._download_semaphore:
            response = await self.attachment_handler.download_image(attachment.url)
            if response:
                return attachment.url

    async def get_thread(self, message: discord.Message) -> discord.Thread | discord.DMChannel:
        """Get or create a thread for a message.

        Args:
            message: The Discord message

        Returns:
            The thread or DM channel for the conversation

        Raises:
            RuntimeError: If thread creation fails
        """
        try:
            # For DM channels, return the channel directly
            if isinstance(message.channel, discord.DMChannel):
                return message.channel

            # For text channels, create or get thread
            if isinstance(message.channel, discord.TextChannel):
                thread_name = f"Chat with {message.author.name}"

                # Try to create a new thread
                try:
                    thread = await message.create_thread(name=thread_name)
                    return thread
                except discord.HTTPException as e:
                    logger.error("Failed to create thread", error=str(e))
                    raise RuntimeError("Failed to create thread") from e

            return message.channel

        except Exception as e:
            logger.error("Error getting/creating thread", error=str(e))
            raise RuntimeError("Failed to get/create thread") from e
