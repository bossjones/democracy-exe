# pylint: disable=no-member
# pylint: disable=possibly-used-before-assignment
# pyright: reportImportCycles=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# mypy: disable-error-code="index"
# mypy: disable-error-code="no-redef"

"""Tests for the DemocracyBot class."""

from __future__ import annotations

import datetime

from collections import Counter
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

import discord
import discord.ext.test as dpytest

from discord import Activity, AllowedMentions, AppInfo, Game, Intents, Status
from discord.ext import commands
from langgraph.graph.state import CompiledStateGraph  # type: ignore[import]

import pytest

import democracy_exe

from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.core.bot import DESCRIPTION, DemocracyBot
from democracy_exe.chatbot.handlers.attachment_handler import AttachmentHandler
from democracy_exe.chatbot.handlers.message_handler import MessageHandler
from democracy_exe.chatbot.utils.guild_utils import preload_guild_data


if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.mark.asyncio
class TestDemocracyBot:
    """Test suite for DemocracyBot class."""

    async def test_init_default_parameters(self, mocker: MockerFixture) -> None:
        """Test initialization with default parameters.

        Args:
            mocker: Pytest mocker fixture
        """
        # Mock ClientSession to prevent actual HTTP session creation
        mocker.patch("aiohttp.ClientSession", return_value=mocker.AsyncMock())
        bot = DemocracyBot()

        # Check default attributes
        assert bot.command_prefix == aiosettings.prefix
        assert bot.description == DESCRIPTION
        assert isinstance(bot.intents, discord.Intents)
        assert bot.intents.message_content is True
        assert bot.intents.guilds is True
        assert bot.intents.members is True
        assert bot.intents.bans is True
        assert bot.intents.emojis is True
        assert bot.intents.voice_states is True
        assert bot.intents.messages is True
        assert bot.intents.reactions is True

        # Check instance attributes
        assert isinstance(bot.command_stats, Counter)
        assert isinstance(bot.socket_stats, Counter)
        assert isinstance(bot.graph, CompiledStateGraph)
        assert isinstance(bot.message_handler, MessageHandler)
        assert isinstance(bot.attachment_handler, AttachmentHandler)
        assert bot.version == democracy_exe.__version__
        assert bot.guild_data == {}
        assert bot.bot_app_info is None
        assert bot.owner_id is None
        assert bot.invite is None
        assert bot.uptime is None

        await bot.close()

    async def test_init_custom_parameters(self, mocker: MockerFixture) -> None:
        """Test initialization with custom parameters.

        Args:
            mocker: Pytest mocker fixture
        """
        # Mock ClientSession to prevent actual HTTP session creation
        mocker.patch("aiohttp.ClientSession", return_value=mocker.AsyncMock())

        custom_prefix = "!"
        custom_description = "Custom bot description"
        custom_intents = discord.Intents.default()
        custom_intents.members = False  # Different from default

        bot = DemocracyBot(command_prefix=custom_prefix, description=custom_description, intents=custom_intents)

        assert bot.command_prefix == custom_prefix
        assert bot.description == custom_description
        assert bot.intents == custom_intents
        assert bot.intents.members is False  # Verify custom intent setting

        await bot.close()

    async def test_get_context(self, bot: DemocracyBot) -> None:
        """Test get_context method.

        Args:
            bot: Bot fixture
        """
        # Create a test message
        message = await dpytest.message("!test")
        context = await bot.get_context(message)

        assert context.bot == bot
        assert context.message == message
        assert context.prefix == "?"  # Default test prefix

    async def test_setup_hook(self, bot: DemocracyBot, mocker: MockerFixture) -> None:
        """Test setup_hook method.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
        """
        # Mock application_info and _load_extensions
        mock_app_info = mocker.AsyncMock(return_value=mocker.Mock(spec=AppInfo))
        mock_load_extensions = mocker.AsyncMock()

        mocker.patch.object(bot, "application_info", mock_app_info)
        mocker.patch.object(bot, "_load_extensions", mock_load_extensions)

        await bot.setup_hook()

        assert bot.prefixes == [aiosettings.prefix]
        assert bot.version == democracy_exe.__version__
        assert isinstance(bot.guild_data, dict)
        mock_app_info.assert_called_once()
        mock_load_extensions.assert_called_once()

    async def test_on_ready(self, bot: DemocracyBot, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
        """Test on_ready event handler.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
            caplog: Pytest log capture fixture
        """
        # Mock the bot.user property and preload_guild_data
        mock_user = mocker.Mock(spec=discord.ClientUser)
        mock_user.id = 123456789
        bot.user = mock_user

        mock_preload = mocker.AsyncMock(return_value={})
        mocker.patch("democracy_exe.chatbot.core.bot.preload_guild_data", mock_preload)

        await bot.on_ready()

        assert bot.invite is not None
        assert bot.invite.startswith("https://discordapp.com/api/oauth2/authorize")
        assert isinstance(bot.uptime, datetime.datetime)
        assert "Ready:" in caplog.text

    async def test_on_message(self, bot: DemocracyBot, mocker: MockerFixture) -> None:
        """Test on_message event handler.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
        """
        # Mock the bot.user property
        mock_user = mocker.Mock(spec=discord.ClientUser)
        mock_user.id = 123456789
        bot.user = mock_user

        # Create a test message that mentions the bot
        message = await dpytest.message(f"<@{bot.user.id}> Hello!")

        # Mock message handler methods
        mock_get_thread = mocker.AsyncMock(return_value=mocker.Mock())
        mock_stream_response = mocker.AsyncMock(return_value="Test response")

        mocker.patch.object(bot.message_handler, "_get_thread", mock_get_thread)
        mocker.patch.object(bot.message_handler, "stream_bot_response", mock_stream_response)

        # Trigger on_message
        await bot.on_message(message)

        # Verify handler methods were called
        mock_get_thread.assert_called_once_with(message)
        mock_stream_response.assert_called_once()

    async def test_on_command_error(self, bot: DemocracyBot, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
        """Test on_command_error event handler.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
            caplog: Pytest log capture fixture
        """
        # Create a mock context
        ctx = mocker.Mock(spec=commands.Context)
        ctx.command = mocker.Mock(qualified_name="test_command")
        ctx.author = mocker.Mock(spec=discord.Member)
        ctx.author.send = mocker.AsyncMock()

        # Test NoPrivateMessage error
        error = commands.NoPrivateMessage()
        await bot.on_command_error(ctx, error)
        ctx.author.send.assert_called_with("This command cannot be used in private messages.")

        # Test DisabledCommand error
        error = commands.DisabledCommand()
        await bot.on_command_error(ctx, error)
        ctx.author.send.assert_called_with("Sorry. This command is disabled and cannot be used.")

        # Test CommandInvokeError
        original_error = Exception("Test error")
        error = commands.CommandInvokeError(original_error)
        await bot.on_command_error(ctx, error)
        assert "In test_command:" in caplog.text

    async def test_close(self, bot: DemocracyBot, mocker: MockerFixture) -> None:
        """Test close method.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
        """
        # Mock the session close method
        mock_session_close = mocker.AsyncMock()
        bot.session.close = mock_session_close

        await bot.close()

        # Verify session was closed
        mock_session_close.assert_called_once()

    async def test_start(self, bot: DemocracyBot, mocker: MockerFixture) -> None:
        """Test start method.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
        """
        # Mock the parent start method
        mock_super_start = mocker.AsyncMock()
        mocker.patch.object(commands.Bot, "start", mock_super_start)

        await bot.start()

        # Verify parent start was called with correct token
        mock_super_start.assert_called_once_with(str(aiosettings.discord_token), reconnect=True)

    async def test_my_background_task(self, bot: DemocracyBot, mocker: MockerFixture) -> None:
        """Test my_background_task method.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
        """
        # Mock necessary methods and attributes
        mock_channel = mocker.Mock(spec=discord.TextChannel)
        mock_channel.send = mocker.AsyncMock()
        mocker.patch.object(bot, "get_channel", return_value=mock_channel)
        mocker.patch.object(bot, "wait_until_ready", mocker.AsyncMock())
        mocker.patch.object(bot, "is_closed", return_value=True)  # Return True to exit the loop immediately

        await bot.my_background_task()

        # Verify message was sent
        mock_channel.send.assert_called_once_with("1")

    async def test_on_worker_monitor(self, bot: DemocracyBot, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
        """Test on_worker_monitor method.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
            caplog: Pytest log capture fixture
        """
        # Mock necessary methods
        mocker.patch.object(bot, "wait_until_ready", mocker.AsyncMock())
        mocker.patch.object(bot, "is_closed", return_value=True)  # Return True to exit the loop immediately

        await bot.on_worker_monitor()

        # Verify log message
        assert "Worker monitor iteration: 1" in caplog.text
