"""Tests for prefix utility functions.

This module contains tests for the Discord bot's prefix handling utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

import structlog

from structlog.testing import capture_logs

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


logger = structlog.get_logger(__name__)


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
    with capture_logs() as captured:
        prefix = get_guild_prefix(mock_bot, 123)
        assert prefix == "!"

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_guild_prefix_existing:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Getting guild prefix" and log.get("guild_id") == 123 and log.get("prefix") == "!"
            for log in captured
        ), "Expected 'Getting guild prefix' message not found in logs"


@pytest.mark.asyncio
async def test_get_guild_prefix_nonexistent(mock_bot: Any, caplog: LogCaptureFixture) -> None:
    """Test getting prefix for a nonexistent guild.

    Args:
        mock_bot: Mock bot fixture
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        # Remove the prefixes attribute to trigger the error
        delattr(mock_bot, "prefixes")
        prefix = get_guild_prefix(mock_bot, 999)
        assert prefix == aiosettings.prefix

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_guild_prefix_nonexistent:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Error getting guild prefix"
            and "Bot has no prefixes attribute" in str(log.get("error"))
            for log in captured
        ), "Expected 'Error getting guild prefix' message not found in logs"


@pytest.mark.asyncio
async def test_get_guild_prefix_error(mock_bot: Any, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
    """Test getting prefix with an error.

    Args:
        mock_bot: Mock bot fixture
        mocker: Pytest mocker fixture
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        # Create a mock dictionary that raises an exception when get is called
        mock_dict = mocker.MagicMock()
        mock_dict.get.side_effect = Exception("Test error")
        mock_bot.prefixes = mock_dict

        prefix = get_guild_prefix(mock_bot, 123)
        assert prefix == aiosettings.prefix

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_guild_prefix_error:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Error getting guild prefix" and "Test error" in str(log.get("error"))
            for log in captured
        ), "Expected 'Error getting guild prefix' message not found in logs"


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
    with capture_logs() as captured:
        mock_message.channel = mocker.MagicMock(spec=["type"])
        mock_message.channel.type = "private"
        mock_message.guild = None

        result = await get_prefix(mock_bot, mock_message)
        assert isinstance(result, list)
        assert any(aiosettings.prefix in prefix for prefix in result)

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_prefix_dm_channel:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Getting prefix for message" and log.get("message_id") == mock_message.id
            for log in captured
        ), "Expected 'Getting prefix for message' message not found in logs"


@pytest.mark.asyncio
async def test_get_prefix_guild_channel(mock_bot: Any, mock_message: Any, mocker: MockerFixture) -> None:
    """Test getting prefix for guild channel.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        mocker: Pytest mocker fixture
    """
    with capture_logs() as captured:
        mock_message.channel = mocker.MagicMock()
        mock_message.guild.id = 123

        result = await get_prefix(mock_bot, mock_message)
        assert isinstance(result, list)
        assert "!" in result[0]

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_prefix_guild_channel:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Getting prefix for message" and log.get("message_id") == mock_message.id
            for log in captured
        ), "Expected 'Getting prefix for message' message not found in logs"


@pytest.mark.asyncio
async def test_get_prefix_multiple_prefixes(mock_bot: Any, mock_message: Any, mocker: MockerFixture) -> None:
    """Test getting prefix for guild with multiple prefixes.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        mocker: Pytest mocker fixture
    """
    with capture_logs() as captured:
        mock_message.channel = mocker.MagicMock()
        mock_message.guild.id = 789  # Guild with multiple prefixes

        result = await get_prefix(mock_bot, mock_message)
        assert isinstance(result, list)
        assert any("!" in prefix for prefix in result)
        assert any("?" in prefix for prefix in result)
        assert any("$" in prefix for prefix in result)

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_prefix_multiple_prefixes:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Getting prefix for message" and log.get("message_id") == mock_message.id
            for log in captured
        ), "Expected 'Getting prefix for message' message not found in logs"


@pytest.mark.asyncio
async def test_get_prefix_when_mentioned(mock_bot: Any, mock_message: Any, mocker: MockerFixture) -> None:
    """Test getting prefix when bot is mentioned.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        mocker: Pytest mocker fixture
    """
    with capture_logs() as captured:
        mock_message.channel = mocker.MagicMock()
        mock_message.guild.id = 123
        mock_message.content = f"<@{mock_bot.user.id}> help"

        result = await get_prefix(mock_bot, mock_message)
        assert isinstance(result, list)
        assert any(str(mock_bot.user.id) in prefix for prefix in result)

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_prefix_when_mentioned:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Getting prefix for message" and log.get("message_id") == mock_message.id
            for log in captured
        ), "Expected 'Getting prefix for message' message not found in logs"


@pytest.mark.asyncio
async def test_get_prefix_error(mock_bot: Any, mock_message: Any, caplog: LogCaptureFixture) -> None:
    """Test getting prefix with an error.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        # Set channel to None to trigger an error
        mock_message.channel = None
        # Also remove the prefixes attribute to ensure we hit the error path
        delattr(mock_bot, "prefixes")
        result = await get_prefix(mock_bot, mock_message)
        assert isinstance(result, list)
        assert any(aiosettings.prefix in prefix for prefix in result)

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_prefix_error:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Error getting prefix" and log.get("error") is not None for log in captured), (
            "Expected 'Error getting prefix' message not found in logs"
        )


def test_prefix_callable_dm(mock_bot: Any, mock_message: Any) -> None:
    """Test prefix callable for DM channel.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
    """
    with capture_logs() as captured:
        mock_message.guild = None
        prefixes = _prefix_callable(mock_bot, mock_message)
        assert isinstance(prefixes, list)
        assert "!" in prefixes
        assert "?" in prefixes
        assert f"<@!{mock_bot.user.id}> " in prefixes
        assert f"<@{mock_bot.user.id}> " in prefixes

        # Debug: Print captured logs
        print("\nCaptured logs in test_prefix_callable_dm:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event") == "Getting prefixes for DM channel" for log in captured), (
            "Expected 'Getting prefixes for DM channel' message not found in logs"
        )


def test_prefix_callable_guild(mock_bot: Any, mock_message: Any) -> None:
    """Test prefix callable for guild channel.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
    """
    with capture_logs() as captured:
        mock_message.guild.id = 123
        prefixes = _prefix_callable(mock_bot, mock_message)
        assert isinstance(prefixes, list)
        assert "!" in prefixes
        assert f"<@!{mock_bot.user.id}> " in prefixes
        assert f"<@{mock_bot.user.id}> " in prefixes

        # Debug: Print captured logs
        print("\nCaptured logs in test_prefix_callable_guild:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Getting prefixes for guild channel" and log.get("guild_id") == 123 for log in captured
        ), "Expected 'Getting prefixes for guild channel' message not found in logs"


def test_prefix_callable_error(mock_bot: Any, mock_message: Any, caplog: LogCaptureFixture) -> None:
    """Test prefix callable with an error.

    Args:
        mock_bot: Mock bot fixture
        mock_message: Mock message fixture
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        mock_bot.user = None
        prefixes = _prefix_callable(mock_bot, mock_message)
        assert isinstance(prefixes, list)
        assert "!" in prefixes
        assert "?" in prefixes

        # Debug: Print captured logs
        print("\nCaptured logs in test_prefix_callable_error:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Error in prefix_callable"
            and "'NoneType' object has no attribute 'id'" in str(log.get("error"))
            for log in captured
        ), "Expected 'Error in prefix_callable' message not found in logs"


@pytest.mark.asyncio
async def test_update_guild_prefix_success(mock_bot: Any, caplog: LogCaptureFixture) -> None:
    """Test updating guild prefix successfully.

    Args:
        mock_bot: Mock bot fixture
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        await update_guild_prefix(mock_bot, 123, "$")
        assert mock_bot.prefixes[123] == ["$"]

        # Debug: Print captured logs
        print("\nCaptured logs in test_update_guild_prefix_success:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Updated prefix for guild"
            and log.get("guild_id") == 123
            and log.get("new_prefix") == "$"
            for log in captured
        ), "Expected 'Updated prefix for guild' message not found in logs"


@pytest.mark.asyncio
async def test_update_guild_prefix_nonexistent(mock_bot: Any, caplog: LogCaptureFixture) -> None:
    """Test updating prefix for nonexistent guild.

    Args:
        mock_bot: Mock bot fixture
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        await update_guild_prefix(mock_bot, 999, "$")
        assert 999 not in mock_bot.prefixes

        # Debug: Print captured logs
        print("\nCaptured logs in test_update_guild_prefix_nonexistent:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Guild not found in prefix cache" and log.get("guild_id") == 999 for log in captured
        ), "Expected 'Guild not found in prefix cache' message not found in logs"


def test_get_prefix_display_error(mock_bot: Any, mocker: MockerFixture, caplog: LogCaptureFixture) -> None:
    """Test getting prefix display with error.

    Args:
        mock_bot: Mock bot fixture
        mocker: Pytest mocker fixture
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        guild = mocker.MagicMock()
        mock_dict = mocker.MagicMock()
        mock_dict.get.side_effect = Exception("Test error")
        mock_bot.prefixes = mock_dict

        display = get_prefix_display(mock_bot, guild)
        assert display == "Default prefixes are: ! ?"

        # Debug: Print captured logs
        print("\nCaptured logs in test_get_prefix_display_error:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(
            log.get("event") == "Error getting prefix display" and "Test error" in str(log.get("error"))
            for log in captured
        ), "Expected 'Error getting prefix display' message not found in logs"
