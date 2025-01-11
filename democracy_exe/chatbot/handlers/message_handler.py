# pyright: reportAttributeAccessIssue=false

"""Message processing and LangGraph integration.

This module contains functionality for processing Discord messages and integrating
with LangGraph for AI responses.
"""
from __future__ import annotations

import asyncio
import re

from typing import Any, Optional, Union, cast

import discord
import structlog

from discord import DMChannel, Message, TextChannel, Thread
from discord.abc import Messageable
from discord.member import Member
from discord.user import User
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph  # type: ignore


logger = structlog.get_logger(__name__)
from PIL import Image

from democracy_exe.chatbot.handlers.attachment_handler import AttachmentHandler
from democracy_exe.chatbot.utils.message_utils import format_inbound_message, get_session_id, prepare_agent_input


class MessageHandler:
    """Handler for processing Discord messages and integrating with LangGraph."""

    def __init__(self, bot: Any) -> None:
        """Initialize the message handler.

        Args:
            bot: The Discord bot instance
        """
        self.bot = bot
        self.attachment_handler = AttachmentHandler()
        self._download_semaphore = asyncio.Semaphore(5)  # Limit concurrent downloads
        self._max_total_size = 50 * 1024 * 1024  # 50MB total limit

    async def check_for_attachments(self, message: discord.Message) -> str:
        """Check and process message attachments.

        Args:
            message: The Discord message

        Returns:
            The processed message content

        Raises:
            ValueError: If attachment processing fails
            RuntimeError: If attachment size exceeds limits
        """
        try:
            content = cast(str, message.content)
            attachments = cast(list[discord.Attachment], message.attachments)

            # Track total size of all attachments
            total_size = 0
            for attachment in attachments:
                total_size += attachment.size
                if total_size > self._max_total_size:
                    logger.error("Attachment size limit exceeded", total_size=total_size, limit=self._max_total_size)
                    raise RuntimeError(f"Total attachment size {total_size} exceeds {self._max_total_size} limit")

            # Handle Tenor GIFs with size limit
            if "https://tenor.com/view/" in content:
                if len(content) > 2048:  # Discord message limit
                    raise ValueError("Message content exceeds Discord limit")
                return await self._handle_tenor_gif(message, content)

            # Handle image URLs with validation and concurrency limit
            image_pattern = r"https?://[^\s<>\"]+?\.(?:png|jpg|jpeg|gif|webp)"
            if re.search(image_pattern, content):
                urls = re.findall(image_pattern, content)
                if len(urls) > 5:  # Limit number of image URLs
                    logger.error("Too many image URLs", count=len(urls), limit=5)
                    raise ValueError("Too many image URLs in message")

                async with self._download_semaphore:
                    return await self._handle_url_image(urls[0])

            # Handle Discord attachments with limits
            if attachments:
                if len(attachments) > 5:  # Limit number of attachments
                    raise ValueError("Too many attachments in message")

                async with self._download_semaphore:
                    return await self._handle_attachment_image(message)

            return content
        except (RuntimeError, ValueError) as e:
            # Re-raise specific errors we want to handle
            raise
        except Exception as e:
            logger.error("Error checking attachments", error=str(e))
            return cast(str, message.content) or ""
        finally:
            # Force cleanup of any large objects
            import gc
            gc.collect()

    async def stream_bot_response(
        self,
        graph: CompiledStateGraph,
        input_data: dict[str, Any]
    ) -> str:
        """Stream responses from the bot's LangGraph.

        Args:
            graph: The compiled state graph
            input_data: Input data for the graph

        Returns:
            The bot's response

        Raises:
            ValueError: If response generation fails
            RuntimeError: If response size exceeds limits or times out
        """
        try:
            # Add timeout for graph processing
            async with asyncio.timeout(30.0):
                # Ensure we await the response if it's a coroutine
                response = await graph.ainvoke(input_data) if hasattr(graph, 'ainvoke') else graph.invoke(input_data)

            if isinstance(response, dict) and "messages" in response:
                messages = response.get("messages", [])
                combined_content = "".join(
                    msg.content for msg in messages
                    if hasattr(msg, 'content')
                )

                # Check response size
                if len(combined_content.encode('utf-8')) > 2000:  # Discord limit
                    logger.error("Response exceeds size limit", size=len(combined_content.encode('utf-8')), limit=2000)
                    raise RuntimeError("Response exceeds Discord message size limit")

                return combined_content

            logger.error("Invalid response format", response=response)
            raise ValueError("No response generated")
        except TimeoutError:
            logger.error("Response generation timed out", timeout=30.0)
            raise RuntimeError("Response generation timed out")
        except Exception as e:
            logger.error("Error streaming bot response", error=str(e))
            raise

    async def _get_thread(self, message: discord.Message) -> Thread | DMChannel:
        """Get or create a thread for the message.

        Args:
            message: The Discord message

        Returns:
            The thread or DM channel for the message

        Raises:
            ValueError: If thread creation fails
        """
        try:
            channel = cast(Union[TextChannel, DMChannel], message.channel)
            if isinstance(channel, DMChannel):
                return channel

            if not isinstance(channel, TextChannel):
                raise ValueError(f"Unsupported channel type: {type(channel)}")

            thread = await channel.create_thread(
                name="Response",
                message=message,
            )
            return thread
        except Exception as e:
            logger.error(f"Error getting thread: {e}")
            raise

    def _format_inbound_message(self, message: Message) -> HumanMessage:
        """Format a Discord message into a HumanMessage.

        Args:
            message: The Discord message to format

        Returns:
            Formatted HumanMessage
        """
        return format_inbound_message(message)

    def get_session_id(self, message: Message | Thread) -> str:
        """Generate a session ID for the given message.

        Args:
            message: The message or thread

        Returns:
            The generated session ID
        """
        return get_session_id(message)

    def prepare_agent_input(
        self,
        message: Message | Thread,
        user_real_name: str,
        surface_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare the agent input from the incoming Discord message.

        Args:
            message: The Discord message containing the user input
            user_real_name: The real name of the user who sent the message
            surface_info: The surface information related to the message

        Returns:
            The input dictionary to be sent to the agent

        Raises:
            ValueError: If message processing fails
        """
        return prepare_agent_input(message, user_real_name, surface_info)

    async def _handle_tenor_gif(self, message: discord.Message, content: str) -> str:
        """Handle Tenor GIF URLs in messages.

        Args:
            message: The Discord message
            content: The message content

        Returns:
            Updated message content with GIF description
        """
        try:
            start_index = content.index("https://tenor.com/view/")
            end_index = content.find(" ", start_index)
            tenor_url = content[start_index:] if end_index == -1 else content[start_index:end_index]

            parts = tenor_url.split("/")
            words = parts[-1].split("-")[:-1]
            sentence = " ".join(words)

            author = cast(Union[Member, User], message.author)
            return f"{content} [{author.display_name} posts an animated {sentence}]".replace(tenor_url, "")
        except Exception as e:
            logger.error(f"Error processing Tenor GIF: {e}")
            return content

    async def _handle_url_image(self, url: str) -> str:
        """Handle image URLs in messages.

        Args:
            url: The image URL

        Returns:
            The original URL

        Raises:
            ValueError: If image processing fails
            RuntimeError: If image size exceeds limits
        """
        try:
            logger.info("Processing image URL", url=url)

            # Add timeout for image download
            async with asyncio.timeout(10.0):
                response = await self.attachment_handler.download_image(url)

            if response:
                # Check image size before processing
                if len(response.getvalue()) > 8 * 1024 * 1024:  # 8MB limit
                    raise RuntimeError("Image size exceeds 8MB limit")

                image = Image.open(response).convert("RGB")
                # Add size limits for image dimensions
                if image.size[0] * image.size[1] > 4096 * 4096:
                    raise RuntimeError("Image dimensions too large")

            return url
        except TimeoutError:
            logger.error("Image download timed out", url=url)
            raise RuntimeError("Image download timed out")
        except Exception as e:
            logger.error("Error processing image URL", error=str(e), url=url)
            return url

    async def _handle_attachment_image(self, message: discord.Message) -> str:
        """Handle image attachments in messages.

        Args:
            message: The Discord message with attachments

        Returns:
            The message content
        """
        try:
            attachments = cast(list[discord.Attachment], message.attachments)
            if not attachments:
                return cast(str, message.content) or ""

            attachment = attachments[0]
            if not attachment.content_type or not attachment.content_type.startswith("image/"):
                return cast(str, message.content) or ""

            response = await self.attachment_handler.download_image(attachment.url)
            if response:
                image = Image.open(response).convert("RGB")
                # Process image if needed

            return cast(str, message.content) or ""
        except Exception as e:
            logger.error(f"Error processing attachment: {e}")
            return cast(str, message.content) or ""
