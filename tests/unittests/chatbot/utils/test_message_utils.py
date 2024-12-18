# pyright: reportAttributeAccessIssue=false
"""Tests for message_utils module."""

from __future__ import annotations

import asyncio

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

import discord

from discord import DMChannel, Guild, Member, Message, TextChannel, Thread, User
from langchain_core.messages import HumanMessage

import pytest

from democracy_exe.chatbot.utils.message_utils import (
    format_inbound_message,
    get_or_create_thread,
    get_session_id,
    prepare_agent_input,
)


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_message(mocker: MockerFixture) -> Message:
    """Create a mock Discord message.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mock Message object
    """
    message = mocker.Mock(spec=Message)
    message.id = "123456789"
    message.content = "Test message content"
    message.author = mocker.Mock(spec=Member)
    message.author.global_name = "TestUser"
    message.author.id = "987654321"
    message.guild = mocker.Mock(spec=Guild)
    message.guild.name = "Test Guild"
    message.channel = mocker.Mock(spec=TextChannel)
    message.channel.id = "456789123"
    message.channel.type = "text"
    message.attachments = []
    return message


@pytest.fixture
def mock_dm_message(mock_message: Message, mocker: MockerFixture) -> Message:
    """Create a mock Discord DM message.

    Args:
        mock_message: Base mock message fixture
        mocker: Pytest mocker fixture

    Returns:
        Mock Message object for DM
    """
    mock_message.guild = None
    mock_message.channel = mocker.Mock(spec=DMChannel)
    mock_message.channel.type = "private"
    return mock_message


@pytest.fixture
def mock_thread(mock_message: Message, mocker: MockerFixture) -> Thread:
    """Create a mock Discord thread.

    Args:
        mock_message: Base mock message fixture
        mocker: Pytest mocker fixture

    Returns:
        Mock Thread object
    """
    thread = mocker.Mock(spec=Thread)
    thread.starter_message = mock_message
    thread.name = "Test Thread"
    return thread


@pytest.mark.asyncio
class TestMessageUtils:
    """Test suite for message_utils module."""

    def test_format_inbound_message_guild(self, mock_message: Message) -> None:
        """Test formatting a guild message.

        Args:
            mock_message: Mock message fixture
        """
        result = format_inbound_message(mock_message)
        assert isinstance(result, HumanMessage)
        assert "Test Guild" in result.content
        assert "Test message content" in result.content
        assert result.name == "TestUser"
        assert result.id == "123456789"

    def test_format_inbound_message_dm(self, mock_dm_message: Message) -> None:
        """Test formatting a DM message.

        Args:
            mock_dm_message: Mock DM message fixture
        """
        result = format_inbound_message(mock_dm_message)
        assert isinstance(result, HumanMessage)
        assert "guild=" not in result.content
        assert "Test message content" in result.content

    def test_format_inbound_message_error(self, mock_message: Message) -> None:
        """Test error handling in message formatting.

        Args:
            mock_message: Mock message fixture
        """
        mock_message.author.global_name = None
        with pytest.raises(ValueError, match="Failed to format message"):
            format_inbound_message(mock_message)

    async def test_get_or_create_thread_dm(self, mock_dm_message: Message) -> None:
        """Test getting/creating thread for DM.

        Args:
            mock_dm_message: Mock DM message fixture
        """
        result = await get_or_create_thread(mock_dm_message)
        assert isinstance(result, DMChannel)

    async def test_get_or_create_thread_text_channel(self, mock_message: Message, mocker: MockerFixture) -> None:
        """Test getting/creating thread for text channel.

        Args:
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        mock_thread = mocker.Mock(spec=Thread)
        mock_message.channel.create_thread.return_value = mock_thread
        result = await get_or_create_thread(mock_message)
        assert result == mock_thread
        mock_message.channel.create_thread.assert_called_once_with(name="Response", message=mock_message)

    async def test_get_or_create_thread_error(self, mock_message: Message, mocker: MockerFixture) -> None:
        """Test error handling in thread creation.

        Args:
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        mock_response = mocker.Mock()
        mock_response.status = 400
        mock_message.channel.create_thread.side_effect = discord.HTTPException(
            response=mock_response, message="Test error"
        )
        with pytest.raises(discord.HTTPException):
            await get_or_create_thread(mock_message)

    def test_get_session_id_dm(self, mock_dm_message: Message) -> None:
        """Test getting session ID for DM.

        Args:
            mock_dm_message: Mock DM message fixture
        """
        result = get_session_id(mock_dm_message)
        assert result == "discord_987654321"

    def test_get_session_id_guild(self, mock_message: Message) -> None:
        """Test getting session ID for guild message.

        Args:
            mock_message: Mock message fixture
        """
        result = get_session_id(mock_message)
        assert result == "discord_456789123"

    def test_get_session_id_thread(self, mock_thread: Thread) -> None:
        """Test getting session ID for thread.

        Args:
            mock_thread: Mock thread fixture
        """
        result = get_session_id(mock_thread)
        assert result == f"discord_{mock_thread.starter_message.channel.id}"

    def test_get_session_id_error(self, mock_message: Message) -> None:
        """Test error handling in session ID generation.

        Args:
            mock_message: Mock message fixture
        """
        mock_message.author.id = None
        result = get_session_id(mock_message)
        assert result.startswith("discord_fallback_")

    def test_prepare_agent_input_message(self, mock_message: Message) -> None:
        """Test preparing agent input from message.

        Args:
            mock_message: Mock message fixture
        """
        surface_info = {"platform": "discord"}
        result = prepare_agent_input(mock_message, "Real Name", surface_info)
        assert result == {
            "user name": "Real Name",
            "message": "Test message content",
            "surface_info": surface_info,
        }

    def test_prepare_agent_input_thread(self, mock_thread: Thread) -> None:
        """Test preparing agent input from thread.

        Args:
            mock_thread: Mock thread fixture
        """
        surface_info = {"platform": "discord"}
        result = prepare_agent_input(mock_thread, "Real Name", surface_info)
        assert result == {
            "user name": "Real Name",
            "message": "Test message content",
            "surface_info": surface_info,
        }

    def test_prepare_agent_input_with_attachments(self, mock_message: Message, mocker: MockerFixture) -> None:
        """Test preparing agent input with attachments.

        Args:
            mock_message: Mock message fixture
            mocker: Pytest mocker fixture
        """
        attachment = mocker.Mock()
        attachment.filename = "test.png"
        attachment.content_type = "image/png"
        attachment.url = "http://test.com/image.png"
        mock_message.attachments = [attachment]

        surface_info = {"platform": "discord"}
        result = prepare_agent_input(mock_message, "Real Name", surface_info)
        assert result == {
            "user name": "Real Name",
            "message": "Test message content",
            "surface_info": surface_info,
            "file_name": "test.png",
            "image_url": "http://test.com/image.png",
        }

    def test_prepare_agent_input_error(self, mock_message: Message) -> None:
        """Test error handling in agent input preparation.

        Args:
            mock_message: Mock message fixture
        """
        mock_message.content = None
        with pytest.raises(ValueError, match="Failed to prepare agent input"):
            prepare_agent_input(mock_message, "Real Name", {})
