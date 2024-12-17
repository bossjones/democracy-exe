"""Unit tests for extension utilities."""

from __future__ import annotations

import os
import pathlib

from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

import aiofiles

from loguru import logger

import pytest

from democracy_exe.chatbot.utils.extension_utils import (
    aio_extensions,
    extensions,
    load_extensions,
    reload_extension,
    unload_extension,
)


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_bot(mocker: MockerFixture) -> Any:
    """Create a mock Discord bot for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Any: A mocked Discord bot object
    """
    mock_b = mocker.Mock()
    mock_b.load_extension = mocker.AsyncMock()
    mock_b.reload_extension = mocker.AsyncMock()
    mock_b.unload_extension = mocker.AsyncMock()
    return mock_b


@pytest.fixture
def mock_cogs_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a mock cogs directory with test files.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        pathlib.Path: Path to the mock cogs directory
    """
    cogs_dir = tmp_path / "cogs"
    cogs_dir.mkdir()

    # Create some test cog files
    (cogs_dir / "__init__.py").touch()
    (cogs_dir / "test_cog1.py").write_text("# Test cog 1")
    (cogs_dir / "test_cog2.py").write_text("# Test cog 2")
    (cogs_dir / "subcogs").mkdir()
    (cogs_dir / "subcogs" / "test_cog3.py").write_text("# Test cog 3")

    return cogs_dir


@pytest.mark.asyncio
class TestExtensionUtils:
    """Test suite for extension utilities."""

    def test_extensions(self, mock_cogs_dir: pathlib.Path, mocker: MockerFixture) -> None:
        """Test synchronous extension discovery.

        Args:
            mock_cogs_dir: Mock cogs directory fixture
            mocker: Pytest mocker fixture
        """
        mocker.patch("democracy_exe.chatbot.utils.extension_utils.HERE", str(mock_cogs_dir.parent))

        ext_list = list(extensions())

        assert len(ext_list) == 3
        assert any("test_cog1" in ext for ext in ext_list)
        assert any("test_cog2" in ext for ext in ext_list)
        assert any("test_cog3" in ext for ext in ext_list)
        assert not any("__init__" in ext for ext in ext_list)

    @pytest.mark.asyncio
    async def test_aio_extensions(self, mock_cogs_dir: pathlib.Path, mocker: MockerFixture) -> None:
        """Test asynchronous extension discovery.

        Args:
            mock_cogs_dir: Mock cogs directory fixture
            mocker: Pytest mocker fixture
        """
        mocker.patch("democracy_exe.chatbot.utils.extension_utils.HERE", str(mock_cogs_dir.parent))

        ext_list = [ext async for ext in aio_extensions()]

        assert len(ext_list) == 3
        assert any("test_cog1" in ext for ext in ext_list)
        assert any("test_cog2" in ext for ext in ext_list)
        assert any("test_cog3" in ext for ext in ext_list)
        assert not any("__init__" in ext for ext in ext_list)

    @pytest.mark.asyncio
    async def test_load_extensions(self, mock_bot: Any, mocker: MockerFixture) -> None:
        """Test loading multiple extensions.

        Args:
            mock_bot: Mock bot fixture
            mocker: Pytest mocker fixture
        """
        extensions_list = ["cogs.test_cog1", "cogs.test_cog2", "cogs.subcogs.test_cog3"]

        await load_extensions(mock_bot, extensions_list)

        assert mock_bot.load_extension.call_count == 3
        for ext in extensions_list:
            mock_bot.load_extension.assert_any_call(ext)

    @pytest.mark.asyncio
    async def test_reload_extension(self, mock_bot: Any) -> None:
        """Test reloading a specific extension.

        Args:
            mock_bot: Mock bot fixture
        """
        extension = "cogs.test_cog1"

        await reload_extension(mock_bot, extension)

        mock_bot.reload_extension.assert_called_once_with(extension)

    @pytest.mark.asyncio
    async def test_unload_extension(self, mock_bot: Any) -> None:
        """Test unloading a specific extension.

        Args:
            mock_bot: Mock bot fixture
        """
        extension = "cogs.test_cog1"

        await unload_extension(mock_bot, extension)

        mock_bot.unload_extension.assert_called_once_with(extension)

    @pytest.mark.asyncio
    async def test_load_extensions_error(self, mock_bot: Any) -> None:
        """Test error handling when loading extensions fails.

        Args:
            mock_bot: Mock bot fixture
        """
        mock_bot.load_extension.side_effect = Exception("Failed to load")

        with pytest.raises(Exception, match="Failed to load"):
            await load_extensions(mock_bot, ["cogs.test_cog1"])

    @pytest.mark.asyncio
    async def test_reload_extension_error(self, mock_bot: Any) -> None:
        """Test error handling when reloading extension fails.

        Args:
            mock_bot: Mock bot fixture
        """
        mock_bot.reload_extension.side_effect = Exception("Failed to reload")

        with pytest.raises(Exception, match="Failed to reload"):
            await reload_extension(mock_bot, "cogs.test_cog1")

    @pytest.mark.asyncio
    async def test_unload_extension_error(self, mock_bot: Any) -> None:
        """Test error handling when unloading extension fails.

        Args:
            mock_bot: Mock bot fixture
        """
        mock_bot.unload_extension.side_effect = Exception("Failed to unload")

        with pytest.raises(Exception, match="Failed to unload"):
            await unload_extension(mock_bot, "cogs.test_cog1")

    def test_extensions_with_invalid_dir(self, mocker: MockerFixture) -> None:
        """Test extension discovery with invalid directory.

        Args:
            mocker: Pytest mocker fixture
        """
        mocker.patch("democracy_exe.chatbot.utils.extension_utils.HERE", "/nonexistent")

        with pytest.raises(Exception):  # noqa: B017
            list(extensions())

    @pytest.mark.asyncio
    async def test_aio_extensions_with_unreadable_file(
        self, mock_cogs_dir: pathlib.Path, mocker: MockerFixture
    ) -> None:
        """Test async extension discovery with unreadable file.

        Args:
            mock_cogs_dir: Mock cogs directory fixture
            mocker: Pytest mocker fixture
        """
        mocker.patch("democracy_exe.chatbot.utils.extension_utils.HERE", str(mock_cogs_dir.parent))
        mocker.patch("aiofiles.open", side_effect=OSError("Permission denied"))

        ext_list = [ext async for ext in aio_extensions()]
        assert len(ext_list) == 0
