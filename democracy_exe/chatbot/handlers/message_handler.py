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
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from PIL import Image

from democracy_exe.chatbot.handlers.attachment_handler import AttachmentHandler


class MessageHandler:
    """Handler for processing Discord messages and integrating with LangGraph."""

    def __init__(self, bot: Any) -> None:
        """Initialize the message handler.

        Args:
            bot: The Discord bot instance
        """
        self.bot = bot
        self.attachment_handler = AttachmentHandler()

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

    def _format_inbound_message(self, message: discord.Message) -> HumanMessage:
        """Format a Discord message as a LangChain HumanMessage.

        Args:
            message: The Discord message

        Returns:
            The formatted HumanMessage
        """
        try:
            content = cast(str, message.content)
            guild = cast(Optional[discord.Guild], message.guild)
            channel = cast(Union[TextChannel, DMChannel], message.channel)
            author = cast(Union[Member, User], message.author)

            if guild:
                content = f"[In server: {guild}, channel: {channel.name}] {content}"

            return HumanMessage(
                content=content,
                name=author.display_name,
                id=str(message.id),
            )
        except Exception as e:
            logger.error(f"Error formatting message: {e}")
            return HumanMessage(content=cast(str, message.content) or "")

    def stream_bot_response(self, graph: CompiledStateGraph, user_input: dict[str, Any]) -> str:
        """Stream responses from the bot's LangGraph.

        Args:
            graph: The compiled LangGraph
            user_input: The user input dictionary

        Returns:
            The concatenated response text
        """
        try:
            response_text = ""
            for output in graph.stream(user_input):
                if "messages" in output:
                    for message in output["messages"]:
                        if isinstance(message, AIMessage):
                            response_text += message.content
            return response_text
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            return "Error generating response"

    async def check_for_attachments(self, message: discord.Message) -> str:
        """Check for and process any attachments in the message.

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

    def get_session_id(self, message: discord.Message | discord.Thread) -> str:
        """Generate a session ID for the given message.

        Args:
            message: The message or event dictionary

        Returns:
            The generated session ID

        Raises:
            ValueError: If message type is not supported
        """
        try:
            if isinstance(message, Thread):
                if not message.starter_message:
                    raise ValueError("Thread has no starter message")
                starter_message = cast(discord.Message, message.starter_message)
                channel = cast(Union[TextChannel, DMChannel], starter_message.channel)
                is_dm = isinstance(channel, DMChannel)
                user_id = starter_message.author.id
                channel_id = channel.id
            elif isinstance(message, discord.Message):
                channel = cast(Union[TextChannel, DMChannel], message.channel)
                is_dm = isinstance(channel, DMChannel)
                user_id = message.author.id
                channel_id = channel.id
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")

            return f"discord_{user_id}" if is_dm else f"discord_{channel_id}"
        except Exception as e:
            logger.error(f"Error generating session ID: {e}")
            return f"discord_error_{id(message)}"

    def prepare_agent_input(
        self, message: discord.Message | discord.Thread, user_name: str, surface_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Prepare input for the agent from a message.

        Args:
            message: The Discord message or thread
            user_name: The user's display name
            surface_info: Additional surface information

        Returns:
            The prepared agent input dictionary
        """
        try:
            if isinstance(message, Thread):
                if not message.starter_message:
                    raise ValueError("Thread has no starter message")
                starter_message = cast(discord.Message, message.starter_message)
                content = starter_message.content
                attachments = cast(list[discord.Attachment], starter_message.attachments)
            else:
                content = cast(str, message.content)
                attachments = cast(list[discord.Attachment], message.attachments)

            agent_input = {
                "user name": user_name,
                "message": content,
                "surface_info": surface_info,
            }

            if attachments:
                attachment = attachments[0]
                agent_input.update({
                    "file_name": attachment.filename,
                    "image_url": attachment.url,
                })

            return agent_input
        except Exception as e:
            logger.error(f"Error preparing agent input: {e}")
            return {
                "user name": user_name,
                "message": "Error processing message",
                "surface_info": surface_info,
            }
