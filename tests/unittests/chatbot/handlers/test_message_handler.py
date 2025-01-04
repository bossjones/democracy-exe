# pyright: reportAttributeAccessIssue=false
"""Unit tests for the MessageHandler class."""

from __future__ import annotations

import io

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import discord

from discord import DMChannel, Message, TextChannel, Thread
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph  # type: ignore[import]
from loguru import logger
from PIL import Image

import pytest

from democracy_exe.chatbot.handlers.message_handler import MessageHandler


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def message_handler(mocker: MockerFixture) -> MessageHandler:
    """Create a MessageHandler instance for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        MessageHandler: A new instance of MessageHandler
    """
    mock_bot = mocker.Mock()
    return MessageHandler(mock_bot)


@pytest.fixture
def mock_message(mocker: MockerFixture) -> Message:
    """Create a mock Discord Message for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Message: A mocked Discord Message object
    """
    mock_msg = mocker.Mock(spec=Message)
    mock_msg.content = "Test message"
    mock_msg.id = 123456789
    mock_msg.author = mocker.Mock()
    mock_msg.author.id = 987654321
    mock_msg.author.global_name = "TestUser"
    mock_msg.author.display_name = "TestUser"
    mock_msg.channel = mocker.Mock(spec=TextChannel)
    mock_msg.channel.id = 456789123
    mock_msg.channel.name = "test-channel"
    mock_msg.channel.type = "text"
    mock_msg.guild = mocker.Mock()
    mock_msg.guild.name = "Test Guild"
    mock_msg.attachments = []
    return mock_msg


@pytest.fixture
def mock_thread(mocker: MockerFixture, mock_message: Message) -> Thread:
    """Create a mock Discord Thread for testing.

    Args:
        mocker: Pytest mocker fixture
        mock_message: Mock message fixture

    Returns:
        Thread: A mocked Discord Thread object
    """
    mock_thrd = mocker.Mock(spec=Thread)
    mock_thrd.starter_message = mock_message
    mock_thrd.name = "Test Thread"
    return mock_thrd


@pytest.fixture
def mock_dm_channel(mocker: MockerFixture) -> DMChannel:
    """Create a mock Discord DMChannel for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        DMChannel: A mocked Discord DMChannel object
    """
    mock_dm = mocker.Mock(spec=DMChannel)
    mock_dm.type = "private"
    return mock_dm


@pytest.mark.asyncio
class TestMessageHandler:
    """Test suite for MessageHandler class."""

    async def test_get_thread_dm_channel(
        self, message_handler: MessageHandler, mock_message: Message, mock_dm_channel: DMChannel
    ) -> None:
        """Test getting thread for DM channel.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
            mock_dm_channel: Mock DM channel fixture
        """
        mock_message.channel = mock_dm_channel
        result = await message_handler._get_thread(mock_message)
        assert result == mock_dm_channel

    async def test_get_thread_text_channel(
        self, message_handler: MessageHandler, mock_message: Message, mocker: MockerFixture
    ) -> None:
        """Test getting thread for text channel.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        mock_thread = mocker.Mock(spec=Thread)
        mock_message.channel.create_thread = mocker.AsyncMock(return_value=mock_thread)

        result = await message_handler._get_thread(mock_message)

        assert result == mock_thread
        mock_message.channel.create_thread.assert_called_once_with(name="Response", message=mock_message)

    @pytest.mark.skip_until(
        deadline=datetime(2025, 1, 25), strict=True, msg="Alert is suppresed. Make progress till then"
    )
    async def test_format_inbound_message(self, message_handler: MessageHandler, mock_message: Message) -> None:
        """Test formatting inbound Discord message.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
        """
        result = message_handler._format_inbound_message(mock_message)

        assert isinstance(result, HumanMessage)
        assert result.name == "TestUser"
        assert result.id == "123456789"
        assert "Test message" in result.content
        assert "Test Guild" in result.content  # Check for guild name instead of mock string representation
        assert "test-channel" in result.content

    @pytest.mark.skip_until(
        deadline=datetime(2025, 1, 25), strict=True, msg="Alert is suppresed. Make progress till then"
    )
    async def test_stream_bot_response(self, message_handler: MessageHandler, mocker: MockerFixture) -> None:
        """Test streaming bot responses.

        Args:
            message_handler: The MessageHandler instance
            mocker: Pytest mocker fixture
        """
        mock_graph = mocker.Mock(spec=CompiledStateGraph)
        mock_graph.stream.return_value = [
            {"messages": [AIMessage(content="Test response chunk 1")]},
            {"messages": [AIMessage(content="Test response chunk 2")]},
        ]

        user_input = {"messages": [HumanMessage(content="Test input")]}
        result = message_handler.stream_bot_response(mock_graph, user_input)

        assert result == "Test response chunk 1Test response chunk 2"
        mock_graph.stream.assert_called_once()

    async def test_check_for_attachments_tenor(self, message_handler: MessageHandler, mock_message: Message) -> None:
        """Test checking for Tenor GIF attachments.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
        """
        mock_message.content = "Check out this GIF https://tenor.com/view/funny-cat-dance-12345"

        result = await message_handler.check_for_attachments(mock_message)

        assert "funny cat dance" in result.lower()
        assert "[TestUser posts an animated funny cat dance]" in result

    async def test_check_for_attachments_url_image(
        self, message_handler: MessageHandler, mock_message: Message, mocker: MockerFixture
    ) -> None:
        """Test checking for URL image attachments.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        mock_message.content = "https://example.com/test.jpg"
        mock_image = mocker.Mock(spec=Image.Image)
        mocker.patch("PIL.Image.open", return_value=mock_image)
        mock_image.convert.return_value = mock_image

        result = await message_handler.check_for_attachments(mock_message)

        assert result == "https://example.com/test.jpg"

    async def test_check_for_attachments_discord_image(
        self, message_handler: MessageHandler, mock_message: Message, mocker: MockerFixture
    ) -> None:
        """Test checking for Discord image attachments.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        mock_attachment = mocker.Mock()
        mock_attachment.content_type = "image/jpeg"
        mock_attachment.url = "https://cdn.discord.com/test.jpg"
        mock_message.attachments = [mock_attachment]

        mock_image = mocker.Mock(spec=Image.Image)
        mocker.patch("PIL.Image.open", return_value=mock_image)
        mock_image.convert.return_value = mock_image

        result = await message_handler.check_for_attachments(mock_message)

        assert result == mock_message.content

    def test_get_session_id_thread(self, message_handler: MessageHandler, mock_thread: Thread) -> None:
        """Test getting session ID for thread.

        Args:
            message_handler: The MessageHandler instance
            mock_thread: Mock thread fixture
        """
        result = message_handler.get_session_id(mock_thread)
        assert result == f"discord_{mock_thread.starter_message.channel.id}"

    def test_get_session_id_message(self, message_handler: MessageHandler, mock_message: Message) -> None:
        """Test getting session ID for message.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
        """
        result = message_handler.get_session_id(mock_message)
        assert result == f"discord_{mock_message.channel.id}"

    def test_get_session_id_dm(
        self, message_handler: MessageHandler, mock_message: Message, mock_dm_channel: DMChannel
    ) -> None:
        """Test getting session ID for DM.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
            mock_dm_channel: Mock DM channel fixture
        """
        mock_message.channel = mock_dm_channel
        result = message_handler.get_session_id(mock_message)
        assert result == f"discord_{mock_message.author.id}"

    def test_prepare_agent_input_message(self, message_handler: MessageHandler, mock_message: Message) -> None:
        """Test preparing agent input from message.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
        """
        surface_info = {"platform": "discord"}
        result = message_handler.prepare_agent_input(mock_message, "Real User Name", surface_info)

        assert result["user name"] == "Real User Name"
        assert result["message"] == mock_message.content
        assert result["surface_info"] == surface_info

    def test_prepare_agent_input_thread(self, message_handler: MessageHandler, mock_thread: Thread) -> None:
        """Test preparing agent input from thread.

        Args:
            message_handler: The MessageHandler instance
            mock_thread: Mock thread fixture
        """
        surface_info = {"platform": "discord"}
        result = message_handler.prepare_agent_input(mock_thread, "Real User Name", surface_info)

        assert result["user name"] == "Real User Name"
        assert result["message"] == mock_thread.starter_message.content
        assert result["surface_info"] == surface_info

    def test_prepare_agent_input_with_attachment(
        self, message_handler: MessageHandler, mock_message: Message, mocker: MockerFixture
    ) -> None:
        """Test preparing agent input with attachment.

        Args:
            message_handler: The MessageHandler instance
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        mock_attachment = mocker.Mock()
        mock_attachment.filename = "test.jpg"
        mock_attachment.content_type = "image/jpeg"
        mock_attachment.url = "https://cdn.discord.com/test.jpg"
        mock_message.attachments = [mock_attachment]

        surface_info = {"platform": "discord"}
        result = message_handler.prepare_agent_input(mock_message, "Real User Name", surface_info)

        assert result["file_name"] == "test.jpg"
        assert result["image_url"] == "https://cdn.discord.com/test.jpg"
