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


@pytest.fixture
async def bot(mocker: MockerFixture) -> AsyncGenerator[DemocracyBot, None]:
    """Create a test bot instance.

    Args:
        mocker: Pytest mocker fixture

    Yields:
        DemocracyBot: A configured test bot instance
    """
    # Mock aiohttp ClientSession
    mock_session = mocker.AsyncMock()
    mocker.patch("aiohttp.ClientSession", return_value=mock_session)

    # Configure intents
    intents = Intents.default()
    intents.message_content = True
    intents.guilds = True
    intents.members = True
    intents.bans = True
    intents.emojis = True
    intents.voice_states = True
    intents.messages = True
    intents.reactions = True

    # Create bot instance
    bot = DemocracyBot(command_prefix="?", intents=intents, description="Test DemocracyBot instance")

    # Mock bot.user property
    mock_user = mocker.Mock(spec=discord.ClientUser)
    mock_user.id = 123456789
    mock_user.__str__ = mocker.Mock(return_value="TestBot")
    mocker.patch.object(bot, "_user", mock_user)

    # Mock websocket
    mock_ws = mocker.AsyncMock()
    mock_ws.open = False
    mocker.patch.object(bot, "ws", mock_ws)

    # Configure test environment
    await bot.setup_hook()
    dpytest.configure(bot)

    yield bot

    # Cleanup
    if hasattr(bot, "session") and not bot.session.closed:
        await bot.session.close()


@pytest.mark.asyncio
class TestDemocracyBot:
    """Test suite for DemocracyBot class."""

    @pytest.mark.asyncio
    async def test_init_default_parameters(self, mocker: MockerFixture) -> None:
        """Test initialization with default parameters.

        Args:
            mocker: Pytest mocker fixture
        """
        # Mock ClientSession to prevent actual HTTP session creation
        mock_session = mocker.AsyncMock()
        mocker.patch("aiohttp.ClientSession", return_value=mock_session)
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

    @pytest.mark.asyncio
    async def test_init_custom_parameters(self, mocker: MockerFixture) -> None:
        """Test initialization with custom parameters.

        Args:
            mocker: Pytest mocker fixture
        """
        # Mock ClientSession to prevent actual HTTP session creation
        mock_session = mocker.AsyncMock()
        mocker.patch("aiohttp.ClientSession", return_value=mock_session)

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

    @pytest.mark.asyncio
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
        assert context.prefix == "?"  # Should match bot's command prefix

    @pytest.mark.asyncio
    async def test_setup_hook(self, bot: DemocracyBot, mocker: MockerFixture) -> None:
        """Test setup_hook method.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
        """
        # Mock application_info and _load_extensions
        mock_app_info = mocker.AsyncMock()
        mock_app_info.return_value = mocker.Mock(spec=AppInfo, owner=mocker.Mock(id=123456789))
        mock_load_extensions = mocker.AsyncMock()

        mocker.patch.object(bot, "application_info", mock_app_info)
        mocker.patch.object(bot, "_load_extensions", mock_load_extensions)

        await bot.setup_hook()

        assert bot.prefixes == [aiosettings.prefix]
        assert bot.version == democracy_exe.__version__
        assert isinstance(bot.guild_data, dict)
        # assert bot.owner_id == 123456789
        assert bot.owner_id
        mock_app_info.assert_called_once()
        mock_load_extensions.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_ready(self, bot: DemocracyBot, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
        """Test on_ready event handler.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
            caplog: Pytest log capture fixture
        """
        # Mock preload_guild_data
        mock_preload = mocker.AsyncMock(return_value={})
        mocker.patch("democracy_exe.chatbot.core.bot.preload_guild_data", mock_preload)

        # Mock bot attributes
        bot.users = [bot.user]
        bot.guilds = [mocker.Mock(spec=discord.Guild)]

        # Call on_ready
        await bot.on_ready()

        # Verify results
        assert (
            bot.invite == f"https://discordapp.com/api/oauth2/authorize?client_id={bot.user.id}&scope=bot&permissions=0"
        )
        assert isinstance(bot.uptime, datetime.datetime)
        assert "Ready: TestBot (ID: 123456789)" in caplog.text

    @pytest.mark.asyncio
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

        # Create a test message with proper author
        message = await dpytest.message(f"<@{bot.user.id}> Hello!")
        message.author = mocker.Mock(spec=discord.Member, id=987654321, bot=False)

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

    @pytest.mark.asyncio
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
        assert f"In {ctx.command.qualified_name}:" in caplog.text

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

        # Mock is_closed() to return True after first iteration
        is_closed_mock = mocker.Mock(side_effect=[False, True])
        mocker.patch.object(bot, "is_closed", is_closed_mock)

        # Mock sleep to avoid actual delay
        mock_sleep = mocker.AsyncMock()
        mocker.patch("asyncio.sleep", mock_sleep)

        await bot.my_background_task()

        # Verify message was sent and sleep was called
        mock_channel.send.assert_called_once_with("1")
        mock_sleep.assert_called_once_with(60)

    @pytest.mark.asyncio
    async def test_on_worker_monitor(self, bot: DemocracyBot, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
        """Test on_worker_monitor method.

        Args:
            bot: Bot fixture
            mocker: Pytest mocker fixture
            caplog: Pytest log capture fixture
        """
        # Mock necessary methods
        mocker.patch.object(bot, "wait_until_ready", mocker.AsyncMock())

        # Mock is_closed() to return True after first iteration
        is_closed_mock = mocker.Mock(side_effect=[False, True])
        mocker.patch.object(bot, "is_closed", is_closed_mock)

        # Mock sleep to avoid actual delay
        mock_sleep = mocker.AsyncMock()
        mocker.patch("asyncio.sleep", mock_sleep)

        # Set log level to capture INFO messages
        caplog.set_level("INFO")

        await bot.on_worker_monitor()

        # Verify log message and sleep was called
        assert "Worker monitor iteration: 1" in caplog.text
        mock_sleep.assert_called_once_with(10)
