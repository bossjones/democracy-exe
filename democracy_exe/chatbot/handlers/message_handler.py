# pyright: reportAttributeAccessIssue=false

"""Message processing and LangGraph integration.

This module contains functionality for processing Discord messages and integrating
with LangGraph for AI responses.
"""
from __future__ import annotations

import re

from typing import Any, Optional, Union, cast

import discord

from discord import DMChannel, Message, TextChannel, Thread
from discord.abc import Messageable
from discord.member import Member
from discord.user import User
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph  # type: ignore
from loguru import logger
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

    async def check_for_attachments(self, message: discord.Message) -> str:
        """Check and process message attachments.

        Args:
            message: The Discord message

        Returns:
            The processed message content
        """
        try:
            content = cast(str, message.content)
            attachments = cast(list[discord.Attachment], message.attachments)

            # Handle Tenor GIFs
            if "https://tenor.com/view/" in content:
                return await self._handle_tenor_gif(message, content)

            # Handle image URLs
            image_pattern = r"https?://[^\s<>\"]+?\.(?:png|jpg|jpeg|gif|webp)"
            if re.search(image_pattern, content):
                url = re.findall(image_pattern, content)[0]
                return await self._handle_url_image(url)

            # Handle Discord attachments
            if attachments:
                return await self._handle_attachment_image(message)

            return content
        except Exception as e:
            logger.error(f"Error checking attachments: {e}")
            return cast(str, message.content) or ""

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
        """
        try:
            response = graph.invoke(input_data)
            if isinstance(response, dict) and "messages" in response:
                messages = response.get("messages", [])
                return "".join(msg.content for msg in messages if hasattr(msg, 'content'))
            raise ValueError("No response generated")
        except Exception as e:
            logger.error(f"Error streaming bot response: {e}")
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
        """
        try:
            logger.info(f"Processing image URL: {url}")
            response = await self.attachment_handler.download_image(url)
            if response:
                image = Image.open(response).convert("RGB")
                # Process image if needed
            return url
        except Exception as e:
            logger.error(f"Error processing image URL: {e}")
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
