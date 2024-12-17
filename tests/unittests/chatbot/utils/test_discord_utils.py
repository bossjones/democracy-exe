# pyright: reportAttributeAccessIssue=false
"""Unit tests for Discord utilities."""

from __future__ import annotations

import os
import pathlib

from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import discord

from discord import DMChannel, Guild, Member, Message, PermissionOverwrite, Role, TextChannel, User
from loguru import logger

import pytest

from democracy_exe.chatbot.utils.discord_utils import (
    create_embed,
    details_from_file,
    filter_empty_string,
    format_user_info,
    get_member_roles_hierarchy,
    get_or_create_role,
    has_required_permissions,
    safe_delete_messages,
    send_chunked_message,
    setup_channel_permissions,
    unlink_orig_file,
)


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_member(mocker: MockerFixture) -> Member:
    """Create a mock Discord Member for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Member: A mocked Discord Member object
    """
    mock_mbr = mocker.Mock(spec=Member)
    mock_mbr.name = "TestUser"
    mock_mbr.id = 123456789
    mock_mbr.nick = "TestNick"
    mock_mbr.created_at = discord.utils.utcnow()
    mock_mbr.joined_at = discord.utils.utcnow()
    mock_mbr.top_role = mocker.Mock(spec=Role, name="TestRole")
    return mock_mbr


@pytest.fixture
def mock_channel(mocker: MockerFixture) -> TextChannel:
    """Create a mock Discord TextChannel for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        TextChannel: A mocked Discord TextChannel object
    """
    mock_ch = mocker.Mock(spec=TextChannel)
    mock_ch.name = "test-channel"
    mock_ch.id = 987654321
    mock_ch.send = mocker.AsyncMock()
    mock_ch.edit = mocker.AsyncMock()
    return mock_ch


@pytest.fixture
def mock_guild(mocker: MockerFixture) -> Guild:
    """Create a mock Discord Guild for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Guild: A mocked Discord Guild object
    """
    mock_g = mocker.Mock(spec=Guild)
    mock_g.name = "Test Guild"
    mock_g.id = 456789123
    mock_g.roles = []
    mock_g.create_role = mocker.AsyncMock()
    return mock_g


@pytest.mark.asyncio
class TestDiscordUtils:
    """Test suite for Discord utilities."""

    def test_has_required_permissions(
        self, mock_member: Member, mock_channel: TextChannel, mocker: MockerFixture
    ) -> None:
        """Test permission checking.

        Args:
            mock_member: Mock member fixture
            mock_channel: Mock channel fixture
            mocker: Pytest mocker fixture
        """
        mock_perms = mocker.Mock()
        mock_perms.send_messages = True
        mock_perms.read_messages = True
        mock_channel.permissions_for.return_value = mock_perms

        result = has_required_permissions(mock_member, mock_channel, {"send_messages", "read_messages"})

        assert result is True
        mock_channel.permissions_for.assert_called_once_with(mock_member)

    @pytest.mark.asyncio
    async def test_send_chunked_message(self, mock_channel: TextChannel) -> None:
        """Test sending chunked messages.

        Args:
            mock_channel: Mock channel fixture
        """
        long_message = "x" * 3000
        messages = await send_chunked_message(mock_channel, long_message, 2000)

        assert len(messages) == 2
        assert mock_channel.send.call_count == 2

    def test_create_embed(self) -> None:
        """Test creating Discord embeds."""
        fields = [{"name": "Field1", "value": "Value1"}, {"name": "Field2", "value": "Value2", "inline": False}]

        embed = create_embed(
            title="Test Title",
            description="Test Description",
            color=discord.Color.blue(),
            fields=fields,
            footer="Test Footer",
            thumbnail_url="https://example.com/image.png",
        )

        assert isinstance(embed, discord.Embed)
        assert embed.title == "Test Title"
        assert embed.description == "Test Description"
        assert len(embed.fields) == 2
        assert embed.footer.text == "Test Footer"
        assert embed.thumbnail.url == "https://example.com/image.png"

    @pytest.mark.asyncio
    @pytest.mark.flaky()
    @pytest.mark.skip(reason="Need to fix this test and make it use dpytest")
    async def test_get_or_create_role_existing(self, mock_guild: Guild, mocker: MockerFixture) -> None:
        """Test getting existing role.

        Args:
            mock_guild: Mock guild fixture
            mocker: Pytest mocker fixture
        """
        existing_role = mocker.Mock(spec=Role, name="TestRole")
        mock_guild.roles = [existing_role]

        mock_guild.create_role = mocker.AsyncMock(return_value=existing_role)
        result = await get_or_create_role(mock_guild, "TestRole")

        assert result == existing_role
        mock_guild.create_role.assert_not_called()
        mock_guild.create_role.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_role_new(self, mock_guild: Guild, mocker: MockerFixture) -> None:
        """Test creating new role.

        Args:
            mock_guild: Mock guild fixture
            mocker: Pytest mocker fixture
        """
        new_role = mocker.Mock(spec=Role, name="NewRole")
        mock_guild.create_role.return_value = new_role

        result = await get_or_create_role(mock_guild, "NewRole", color=discord.Color.blue())

        assert result == new_role
        mock_guild.create_role.assert_called_once_with(name="NewRole", color=discord.Color.blue())

    @pytest.mark.asyncio
    async def test_safe_delete_messages(self, mocker: MockerFixture) -> None:
        """Test safe message deletion.

        Args:
            mocker: Pytest mocker fixture
        """
        mock_messages = [mocker.Mock(spec=Message, id=i) for i in range(3)]
        for msg in mock_messages:
            msg.delete = mocker.AsyncMock()

        await safe_delete_messages(mock_messages, delay=0.1)

        for msg in mock_messages:
            msg.delete.assert_called_once()

    def test_get_member_roles_hierarchy(self, mock_member: Member, mocker: MockerFixture) -> None:
        """Test getting member roles hierarchy.

        Args:
            mock_member: Mock member fixture
            mocker: Pytest mocker fixture
        """
        roles = [mocker.Mock(spec=Role, position=i) for i in range(3)]
        mock_member.roles = roles

        result = get_member_roles_hierarchy(mock_member)

        assert len(result) == 3
        assert result[0].position == 2  # Highest position first
        assert result[-1].position == 0  # Lowest position last

    @pytest.mark.asyncio
    async def test_setup_channel_permissions(self, mock_channel: TextChannel, mocker: MockerFixture) -> None:
        """Test setting up channel permissions.

        Args:
            mock_channel: Mock channel fixture
            mocker: Pytest mocker fixture
        """
        role = mocker.Mock(spec=Role)
        overwrites = {role: mocker.Mock(spec=PermissionOverwrite)}

        await setup_channel_permissions(mock_channel, overwrites)

        mock_channel.edit.assert_called_once_with(overwrites=overwrites)

    def test_format_user_info_user(self, mocker: MockerFixture) -> None:
        """Test formatting user information.

        Args:
            mocker: Pytest mocker fixture
        """
        mock_user = mocker.Mock(spec=User)
        mock_user.name = "TestUser"
        mock_user.id = 123456789
        mock_user.created_at = discord.utils.utcnow()

        result = format_user_info(mock_user)

        assert "TestUser" in result
        assert "123456789" in result
        assert "Created:" in result

    def test_format_user_info_member(self, mock_member: Member) -> None:
        """Test formatting member information.

        Args:
            mock_member: Mock member fixture
        """
        result = format_user_info(mock_member)

        assert "TestUser" in result
        assert "123456789" in result
        assert "TestNick" in result
        assert "TestRole" in result
        assert "Created:" in result
        assert "Joined:" in result

    @pytest.mark.asyncio
    async def test_details_from_file(self, tmp_path: pathlib.Path, mocker: MockerFixture) -> None:
        """Test getting file details.

        Args:
            tmp_path: Pytest temporary directory fixture
            mocker: Pytest mocker fixture
        """
        test_file = tmp_path / "test.mp4"
        test_file.write_text("test content")

        mocker.patch(
            "democracy_exe.shell._aio_run_process_and_communicate", return_value="2024-12-17 12:47:24.633184878 -0500"
        )

        input_file, output_file, timestamp = await details_from_file(str(test_file))

        assert input_file == "test.mp4"
        assert output_file == "test_smaller.mp4"
        assert timestamp == "2024-12-17 14:28:55"

    def test_filter_empty_string(self) -> None:
        """Test filtering empty strings from list."""
        test_list = ["", "test1", "", "test2", ""]
        result = filter_empty_string(test_list)

        assert len(result) == 2
        assert "" not in result
        assert "test1" in result
        assert "test2" in result
