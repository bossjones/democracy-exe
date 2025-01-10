"""Unit tests for guild utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import structlog


logger = structlog.get_logger(__name__)

import pytest

from democracy_exe.chatbot.utils.guild_utils import preload_guild_data
from democracy_exe.factories import guild_factory


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_guild(mocker: MockerFixture) -> Any:
    """Create a mock Guild for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Any: A mocked Guild object
    """
    mock_g = mocker.Mock()
    mock_g.id = 123456789
    mock_g.prefix = "!"
    return mock_g


@pytest.mark.asyncio
class TestGuildUtils:
    """Test suite for guild utilities."""

    async def test_preload_guild_data(self, mocker: MockerFixture, mock_guild: Any) -> None:
        """Test preloading guild data.

        Args:
            mocker: Pytest mocker fixture
            mock_guild: Mock guild fixture
        """
        mocker.patch.object(guild_factory, "Guild", return_value=mock_guild)

        result = await preload_guild_data()

        assert isinstance(result, dict)
        assert len(result) == 1
        assert mock_guild.id in result
        assert result[mock_guild.id]["prefix"] == mock_guild.prefix

    async def test_preload_guild_data_multiple(self, mocker: MockerFixture) -> None:
        """Test preloading data for multiple guilds.

        Args:
            mocker: Pytest mocker fixture
        """
        mock_guilds = [mocker.Mock(id=i, prefix=f"!{i}") for i in range(3)]
        mocker.patch.object(guild_factory, "Guild", side_effect=mock_guilds)

        result = await preload_guild_data()

        assert isinstance(result, dict)
        assert len(result) == 3
        for i, guild in enumerate(mock_guilds):
            assert guild.id in result
            assert result[guild.id]["prefix"] == guild.prefix

    async def test_preload_guild_data_empty(self, mocker: MockerFixture) -> None:
        """Test preloading guild data with no guilds.

        Args:
            mocker: Pytest mocker fixture
        """
        mocker.patch.object(guild_factory, "Guild", side_effect=[])

        result = await preload_guild_data()

        assert isinstance(result, dict)
        assert len(result) == 0

    async def test_preload_guild_data_error(self, mocker: MockerFixture) -> None:
        """Test error handling during guild data preloading.

        Args:
            mocker: Pytest mocker fixture
        """
        mocker.patch.object(guild_factory, "Guild", side_effect=Exception("Failed to create guild"))

        result = await preload_guild_data()

        assert isinstance(result, dict)
        assert len(result) == 0

    async def test_preload_guild_data_with_custom_prefix(self, mocker: MockerFixture) -> None:
        """Test preloading guild data with custom prefix.

        Args:
            mocker: Pytest mocker fixture
        """
        mock_guild = mocker.Mock(id=123, prefix="$")
        mocker.patch.object(guild_factory, "Guild", return_value=mock_guild)

        result = await preload_guild_data()

        assert isinstance(result, dict)
        assert len(result) == 1
        assert result[mock_guild.id]["prefix"] == "$"
