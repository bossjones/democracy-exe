# pyright: reportAttributeAccessIssue=false
"""Unit tests for LangChain utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import discord
import structlog

from discord import DMChannel, Guild, Message, TextChannel, Thread
from langchain_core.messages import AIMessage, HumanMessage


logger = structlog.get_logger(__name__)

import pytest

from democracy_exe.chatbot.ai.langchain_utils import (
    format_inbound_message,
    get_session_id,
    get_thread,
    prepare_agent_input,
    stream_bot_response,
)


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


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
    mock_msg.guild = mocker.Mock(spec=Guild)
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
class TestLangChainUtils:
    """Test suite for LangChain utilities."""

    def test_format_inbound_message(self, mock_message: Message) -> None:
        """Test formatting inbound Discord message.

        Args:
            mock_message: Mock message fixture
        """
        result = format_inbound_message(mock_message)

        assert isinstance(result, HumanMessage)
        assert result.name == "TestUser"
        assert result.id == "123456789"
        assert "Test message" in result.content
        assert "Test Guild" in result.content
        assert str(mock_message.channel) in result.content

    def test_format_inbound_message_no_guild(self, mock_message: Message) -> None:
        """Test formatting inbound message without guild.

        Args:
            mock_message: Mock message fixture
        """
        mock_message.guild = None
        result = format_inbound_message(mock_message)

        assert isinstance(result, HumanMessage)
        assert "guild=" not in result.content

    @pytest.mark.flaky()
    @pytest.mark.skip(reason="Need to fix this test to work with pytest-recording")
    def test_stream_bot_response(self, mocker: MockerFixture) -> None:
        """Test streaming bot responses.

        Args:
            mocker: Pytest mocker fixture
        """
        mock_graph = mocker.Mock()
        mock_graph.stream.return_value = [
            {"messages": [AIMessage(content="Test response chunk 1")]},
            {"messages": [AIMessage(content="Test response chunk 2")]},
        ]

        user_input = {"messages": [HumanMessage(content="Test input")]}
        result = stream_bot_response(mock_graph, user_input)

        assert result == "Test response chunk 1Test response chunk 2"
        mock_graph.stream.assert_called_once()

    @pytest.mark.flaky()
    @pytest.mark.skip(reason="Need to fix this test to work with pytest-recording")
    def test_stream_bot_response_empty(self, mocker: MockerFixture) -> None:
        """Test streaming bot responses with empty response.

        Args:
            mocker: Pytest mocker fixture
        """
        mock_graph = mocker.Mock()
        mock_graph.stream.return_value = [{"messages": []}]

        user_input = {"messages": [HumanMessage(content="Test input")]}
        result = stream_bot_response(mock_graph, user_input)

        assert result == ""
        mock_graph.stream.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_thread_dm_channel(self, mock_message: Message, mock_dm_channel: DMChannel) -> None:
        """Test getting thread for DM channel.

        Args:
            mock_message: Mock message fixture
            mock_dm_channel: Mock DM channel fixture
        """
        mock_message.channel = mock_dm_channel
        result = await get_thread(mock_message)
        assert result == mock_dm_channel

    @pytest.mark.asyncio
    async def test_get_thread_text_channel(self, mock_message: Message, mocker: MockerFixture) -> None:
        """Test getting thread for text channel.

        Args:
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        mock_thread = mocker.Mock(spec=Thread)
        mock_message.channel.create_thread = mocker.AsyncMock(return_value=mock_thread)

        result = await get_thread(mock_message)

        assert result == mock_thread
        mock_message.channel.create_thread.assert_called_once_with(name="Response", message=mock_message)

    def test_prepare_agent_input_message(self, mock_message: Message) -> None:
        """Test preparing agent input from message.

        Args:
            mock_message: Mock message fixture
        """
        surface_info = {"platform": "discord"}
        result = prepare_agent_input(mock_message, "Real User Name", surface_info)

        assert result["user name"] == "Real User Name"
        assert result["message"] == mock_message.content
        assert result["surface_info"] == surface_info

    def test_prepare_agent_input_thread(self, mock_thread: Thread) -> None:
        """Test preparing agent input from thread.

        Args:
            mock_thread: Mock thread fixture
        """
        surface_info = {"platform": "discord"}
        result = prepare_agent_input(mock_thread, "Real User Name", surface_info)

        assert result["user name"] == "Real User Name"
        assert result["message"] == mock_thread.starter_message.content
        assert result["surface_info"] == surface_info

    def test_prepare_agent_input_with_attachment(self, mock_message: Message, mocker: MockerFixture) -> None:
        """Test preparing agent input with attachment.

        Args:
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        mock_attachment = mocker.Mock()
        mock_attachment.filename = "test.jpg"
        mock_attachment.content_type = "image/jpeg"
        mock_attachment.url = "https://cdn.discord.com/test.jpg"
        mock_message.attachments = [mock_attachment]

        surface_info = {"platform": "discord"}
        result = prepare_agent_input(mock_message, "Real User Name", surface_info)

        assert result["file_name"] == "test.jpg"
        assert result["image_url"] == "https://cdn.discord.com/test.jpg"

    @pytest.mark.flaky()
    @pytest.mark.skip(reason="Need to fix this test")
    def test_get_session_id_thread(self, mock_thread: Thread) -> None:
        """Test getting session ID for thread.

        Args:
            mock_thread: Mock thread fixture
        """
        result = get_session_id(mock_thread)
        assert result == f"discord_{mock_thread.starter_message.channel.id}"

    def test_get_session_id_message(self, mock_message: Message) -> None:
        """Test getting session ID for message.

        Args:
            mock_message: Mock message fixture
        """
        result = get_session_id(mock_message)
        assert result == f"discord_{mock_message.channel.id}"

    def test_get_session_id_dm(self, mock_message: Message, mock_dm_channel: DMChannel) -> None:
        """Test getting session ID for DM.

        Args:
            mock_message: Mock message fixture
            mock_dm_channel: Mock DM channel fixture
        """
        mock_message.channel = mock_dm_channel
        result = get_session_id(mock_message)
        assert result == f"discord_{mock_message.author.id}"

    def test_get_session_id_fallback(self, mock_message: Message, mocker: MockerFixture) -> None:
        """Test getting session ID with fallback.

        Args:
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        mock_message.channel = None  # Force an error
        mock_timestamp = 1234567890.0
        mocker.patch("discord.utils.utcnow").return_value.timestamp.return_value = mock_timestamp

        result = get_session_id(mock_message)
        assert result == f"discord_fallback_{mock_timestamp}"
