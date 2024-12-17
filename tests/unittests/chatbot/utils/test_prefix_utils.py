"""Tests for prefix utility functions.

This module contains tests for the Discord bot's prefix handling utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import pytest

from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.utils.prefix_utils import (
    _prefix_callable,
    get_guild_prefix,
    get_prefix,
    get_prefix_display,
    update_guild_prefix,
)


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_bot(mocker: MockerFixture) -> Any:
    """Create a mock bot instance with prefixes.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mock bot instance
    """
    bot = mocker.MagicMock()
    bot.prefixes = {
        123: ["!"],
        456: ["?"],
        789: ["!", "?", "$"],  # Multiple prefixes
    }
    bot.user.id = 789
    return bot


@pytest.fixture
def mock_message(mocker: MockerFixture) -> Any:
    """Create a mock message instance.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mock message instance
    """
    message = mocker.MagicMock()
    message.id = 999
    return message


@pytest.mark.asyncio
async def test_get_guild_prefix_existing(mock_bot: Any, caplog: LogCaptureFixture) -> None:
    """Test getting prefix for an existing guild.

    Args:
        mock_bot: Mock bot fixture
        caplog: Pytest log capture fixture
    """
    prefix = get_guild_prefix(mock_bot, 123)
    assert prefix == "!"
    assert "Error getting guild prefix" not in caplog.text


@pytest.mark.asyncio
async def test_get_guild_prefix_nonexistent(mock_bot: Any, caplog: LogCaptureFixture) -> None:
    """Test getting prefix for a nonexistent guild.

    Args:
        mock_bot: Mock bot fixture
        caplog: Pytest log capture fixture
    """
    prefix = get_guild_prefix(mock_bot, 999)
    assert prefix == aiosettings.prefix
    assert "Error getting guild prefix" not in caplog.text


@pytest.mark.asyncio
async def test_get_guild_prefix_error(mock_bot: Any, caplog: LogCaptureFixture) -> None:
    """Test getting prefix with an error.

    Args:
        mock_bot: Mock bot fixture
        caplog: Pytest log capture fixture
    """
    mock_bot.prefixes.get.side_effect = Exception("Test error")
    prefix = get_guild_prefix(mock_bot, 123)
    assert prefix == aiosettings.prefix
    assert "Error getting guild prefix: Test error" in caplog.text


@pytest.mark.asyncio
async def test_get_prefix_dm_channel(
    mock_bot: Any, mock_message: Any, mocker: MockerFixture, caplog: LogCaptureFixture
) -> None:
    """Test getting prefix for DM channel.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        mocker: Pytest mocker fixture
        caplog: Pytest log capture fixture
    """
    mock_message.channel = mocker.MagicMock(spec=["type"])
    mock_message.channel.type = "private"
    mock_message.guild = None

    result = await get_prefix(mock_bot, mock_message)
    assert isinstance(result, list)
    assert aiosettings.prefix in result[0]
    assert f"Getting prefix for message: {mock_message.id}" in caplog.text


@pytest.mark.asyncio
async def test_get_prefix_guild_channel(mock_bot: Any, mock_message: Any, mocker: MockerFixture) -> None:
    """Test getting prefix for guild channel.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        mocker: Pytest mocker fixture
    """
    mock_message.channel = mocker.MagicMock()
    mock_message.guild.id = 123

    result = await get_prefix(mock_bot, mock_message)
    assert isinstance(result, list)
    assert "!" in result[0]


@pytest.mark.asyncio
async def test_get_prefix_multiple_prefixes(mock_bot: Any, mock_message: Any, mocker: MockerFixture) -> None:
    """Test getting prefix for guild with multiple prefixes.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        mocker: Pytest mocker fixture
    """
    mock_message.channel = mocker.MagicMock()
    mock_message.guild.id = 789  # Guild with multiple prefixes

    result = await get_prefix(mock_bot, mock_message)
    assert isinstance(result, list)
    assert any("!" in prefix for prefix in result)
    assert any("?" in prefix for prefix in result)
    assert any("$" in prefix for prefix in result)


@pytest.mark.asyncio
async def test_get_prefix_when_mentioned(mock_bot: Any, mock_message: Any, mocker: MockerFixture) -> None:
    """Test getting prefix when bot is mentioned.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        mocker: Pytest mocker fixture
    """
    mock_message.channel = mocker.MagicMock()
    mock_message.guild.id = 123
    mock_message.content = f"<@{mock_bot.user.id}> help"

    result = await get_prefix(mock_bot, mock_message)
    assert isinstance(result, list)
    assert any(str(mock_bot.user.id) in prefix for prefix in result)


@pytest.mark.asyncio
async def test_get_prefix_error(mock_bot: Any, mock_message: Any, caplog: LogCaptureFixture) -> None:
    """Test getting prefix with an error.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        caplog: Pytest log capture fixture
    """
    mock_message.channel = None
    result = await get_prefix(mock_bot, mock_message)
    assert isinstance(result, list)
    assert aiosettings.prefix in result[0]
    assert "Error getting prefix" in caplog.text


def test_prefix_callable_dm(mock_bot: Any, mock_message: Any) -> None:
    """Test prefix callable for DM channel.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
    """
    mock_message.guild = None
    prefixes = _prefix_callable(mock_bot, mock_message)
    assert isinstance(prefixes, list)
    assert "!" in prefixes
    assert "?" in prefixes
    assert f"<@!{mock_bot.user.id}> " in prefixes
    assert f"<@{mock_bot.user.id}> " in prefixes


def test_prefix_callable_guild(mock_bot: Any, mock_message: Any) -> None:
    """Test prefix callable for guild channel.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
    """
    mock_message.guild.id = 123
    prefixes = _prefix_callable(mock_bot, mock_message)
    assert isinstance(prefixes, list)
    assert "!" in prefixes
    assert f"<@!{mock_bot.user.id}> " in prefixes
    assert f"<@{mock_bot.user.id}> " in prefixes


def test_prefix_callable_error(mock_bot: Any, mock_message: Any, caplog: LogCaptureFixture) -> None:
    """Test prefix callable with an error.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        caplog: Pytest log capture fixture
    """
    mock_bot.user = None
    prefixes = _prefix_callable(mock_bot, mock_message)
    assert isinstance(prefixes, list)
    assert "!" in prefixes
    assert "?" in prefixes
    assert "Error in prefix_callable" in caplog.text


@pytest.mark.asyncio
async def test_update_guild_prefix_success(mock_bot: Any, caplog: LogCaptureFixture) -> None:
    """Test updating guild prefix successfully.

    Args:
        mock_bot: Mock bot fixture
        caplog: Pytest log capture fixture
    """
    await update_guild_prefix(mock_bot, 123, "$")
    assert mock_bot.prefixes[123] == ["$"]
    assert "Updated prefix for guild 123 to $" in caplog.text


@pytest.mark.asyncio
async def test_update_guild_prefix_invalid(mock_bot: Any) -> None:
    """Test updating guild prefix with invalid prefix.

    Args:
        mock_bot: Mock bot fixture
    """
    with pytest.raises(ValueError):
        await update_guild_prefix(mock_bot, 123, "")

    with pytest.raises(ValueError):
        await update_guild_prefix(mock_bot, 123, "a" * 11)


@pytest.mark.asyncio
async def test_update_guild_prefix_nonexistent(mock_bot: Any, caplog: LogCaptureFixture) -> None:
    """Test updating prefix for nonexistent guild.

    Args:
        mock_bot: Mock bot fixture
        caplog: Pytest log capture fixture
    """
    await update_guild_prefix(mock_bot, 999, "$")
    assert 999 not in mock_bot.prefixes
    assert "Guild 999 not found in prefix cache" in caplog.text


def test_get_prefix_display_dm(mock_bot: Any) -> None:
    """Test getting prefix display for DM.

    Args:
        mock_bot: Mock bot fixture
    """
    display = get_prefix_display(mock_bot)
    assert "Current prefixes are: !" in display
    assert "?" in display


def test_get_prefix_display_guild(mock_bot: Any, mocker: MockerFixture) -> None:
    """Test getting prefix display for guild.

    Args:
        mock_bot: Mock bot fixture
        mocker: Pytest mocker fixture
    """
    guild = mocker.MagicMock()
    guild.id = 123
    display = get_prefix_display(mock_bot, guild)
    assert display == "Current prefix is: !"


def test_get_prefix_display_multiple_prefixes(mock_bot: Any, mocker: MockerFixture) -> None:
    """Test getting prefix display for guild with multiple prefixes.

    Args:
        mock_bot: Mock bot fixture
        mocker: Pytest mocker fixture
    """
    guild = mocker.MagicMock()
    guild.id = 789  # Guild with multiple prefixes
    display = get_prefix_display(mock_bot, guild)
    assert "Current prefixes are: !" in display
    assert "?" in display
    assert "$" in display


def test_get_prefix_display_error(mock_bot: Any, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
    """Test getting prefix display with error.

    Args:
        mock_bot: Mock bot fixture
        mocker: Pytest mocker fixture
        caplog: Pytest log capture fixture
    """
    guild = mocker.MagicMock()
    mock_bot.prefixes.get.side_effect = Exception("Test error")
    display = get_prefix_display(mock_bot, guild)
    assert display == "Default prefixes are: ! ?"
    assert "Error getting prefix display: Test error" in caplog.text
