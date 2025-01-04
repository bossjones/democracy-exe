# pyright: reportAttributeAccessIssue=false
"""Tests for Autocrop cog functionality."""

from __future__ import annotations

import asyncio
import datetime
import logging
import pathlib
import sys

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

import discord
import discord.ext.test as dpytest
import pytest_asyncio

from loguru import logger

import pytest

from democracy_exe.chatbot.cogs.autocrop import HELP_MESSAGE, AutocropError
from democracy_exe.chatbot.cogs.autocrop import Autocrop as AutocropCog
from democracy_exe.chatbot.core.bot import DemocracyBot
from democracy_exe.utils._testing import ContextLogger


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.logging import LogCaptureFixture

    from pytest_mock.plugin import MockerFixture


@pytest_asyncio.fixture(autouse=True)
async def bot_with_autocrop_cog() -> AsyncGenerator[DemocracyBot, None]:
    """Create a DemocracyBot instance for testing.

    Returns:
        AsyncGenerator[DemocracyBot, None]: DemocracyBot instance with test configuration
    """
    intents = discord.Intents.all()
    intents.message_content = True
    intents.guilds = True
    intents.members = True
    intents.bans = True
    intents.emojis = True
    intents.voice_states = True
    intents.messages = True
    intents.reactions = True

    test_bot = DemocracyBot(command_prefix="?", intents=intents, description="Test DemocracyBot instance")
    await test_bot._async_setup_hook()

    test_autocrop_cog = AutocropCog(test_bot)
    await test_bot.add_cog(test_autocrop_cog)
    dpytest.configure(test_bot)

    yield test_bot

    try:
        await dpytest.empty_queue()
    finally:
        pass


@pytest_asyncio.fixture(autouse=True)
async def cleanup_global_message_queue() -> AsyncGenerator[None, None]:
    """Clean up the global message queue after each test."""
    yield
    try:
        await dpytest.empty_queue()
    finally:
        pass


@pytest.fixture
def test_image(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a test image file.

    Args:
        tmp_path: Temporary directory path

    Returns:
        pathlib.Path: Path to test image
    """
    import numpy as np

    from PIL import Image

    # Create a simple test image
    img_array = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    # Save to temp directory
    img_path = tmp_path / "test_image.png"
    img.save(img_path)

    return img_path


@pytest.mark.asyncio
async def test_autocrop_cog_init(bot_with_autocrop_cog: DemocracyBot) -> None:
    """Test Autocrop cog initialization.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
    """
    cog = bot_with_autocrop_cog.get_cog("Autocrop")
    assert isinstance(cog, AutocropCog)
    assert isinstance(cog.bot, discord.ext.commands.Bot)


@pytest.mark.asyncio
async def test_autocrop_cog_on_ready(bot_with_autocrop_cog: DemocracyBot, caplog: LogCaptureFixture) -> None:
    """Test Autocrop cog on_ready event.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        caplog: Pytest log capture fixture
    """
    cog = bot_with_autocrop_cog.get_cog("Autocrop")
    await cog.on_ready()
    assert any(record.message == "Autocrop Cog ready." for record in caplog.records)


@pytest.mark.asyncio
async def test_autocrop_cog_on_guild_join(bot_with_autocrop_cog: DemocracyBot, caplog: LogCaptureFixture) -> None:
    """Test Autocrop cog on_guild_join event.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        caplog: Pytest log capture fixture
    """
    cog = bot_with_autocrop_cog.get_cog("Autocrop")
    guild = bot_with_autocrop_cog.guilds[0]
    await cog.on_guild_join(guild)
    assert any(f"Adding new guild to database: {guild.id}" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_crop_help_command(bot_with_autocrop_cog: DemocracyBot) -> None:
    """Test crop help command shows help message.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
    """
    await dpytest.message("?crop")
    assert dpytest.verify().message().content(HELP_MESSAGE)


@pytest.mark.asyncio
async def test_process_image(
    bot_with_autocrop_cog: DemocracyBot,
    test_image: pathlib.Path,
    tmp_path: pathlib.Path,
    caplog: LogCaptureFixture,
) -> None:
    """Test image processing functionality.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        test_image: Test image fixture
        tmp_path: Temporary directory path
        caplog: Pytest log capture fixture
    """
    cog = bot_with_autocrop_cog.get_cog("Autocrop")
    output_path = tmp_path / "output.png"

    # Test square crop
    success = await cog._process_image(str(test_image), (1, 1), str(output_path))
    assert success

    # Verify output image dimensions
    from PIL import Image

    with Image.open(output_path) as img:
        width, height = img.size
        assert width == height  # Should be square


@pytest.mark.asyncio
async def test_crop_invalid_attachment(bot_with_autocrop_cog: DemocracyBot) -> None:
    """Test crop command with invalid attachment.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
    """
    await dpytest.message("?crop square")
    assert dpytest.verify().message().content("Please attach an image to crop")


@pytest.fixture
def invalid_file(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create an invalid file for testing.

    Args:
        tmp_path: Temporary directory path

    Returns:
        pathlib.Path: Path to invalid test file
    """
    # Create a text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is not an image")
    return test_file


@pytest.mark.asyncio
async def test_crop_invalid_file_type(
    bot_with_autocrop_cog: DemocracyBot,
    invalid_file: pathlib.Path,
) -> None:
    """Test crop command with invalid file type.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        invalid_file: Path to invalid test file
    """
    # Send command with invalid file attachment
    await dpytest.message("?crop square", attachments=[invalid_file])

    # Verify error message
    assert dpytest.verify().message().content("Please provide a valid image file")


@pytest.mark.asyncio
async def test_crop_download_timeout(
    bot_with_autocrop_cog: DemocracyBot, mocker: MockerFixture, test_image: pathlib.Path
) -> None:
    """Test image download timeout handling.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        mocker: Pytest mocker fixture
        test_image: Test image fixture
    """

    # Mock attachment.save to simulate timeout
    async def slow_save(*args, **kwargs):
        await asyncio.sleep(aiosettings.autocrop_download_timeout + 1)

    mock_attachment = mocker.Mock(spec=discord.Attachment)
    mock_attachment.save.side_effect = slow_save
    mock_attachment.content_type = "image/png"
    mock_attachment.filename = "test.png"

    cog = bot_with_autocrop_cog.get_cog("Autocrop")
    success, error_msg = await cog._handle_crop(
        await dpytest.get_message().channel.send("test"),  # Get context
        "square",
        mock_attachment,
    )

    assert not success
    assert "Image download timed out" in error_msg


@pytest.mark.asyncio
async def test_crop_processing_timeout(
    bot_with_autocrop_cog: DemocracyBot, mocker: MockerFixture, test_image: pathlib.Path
) -> None:
    """Test image processing timeout handling.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        mocker: MockerFixture
        test_image: Test image fixture
    """

    # Mock _process_image to simulate timeout
    async def slow_process(*args, **kwargs):
        await asyncio.sleep(aiosettings.autocrop_processing_timeout + 1)
        return True

    cog = bot_with_autocrop_cog.get_cog("Autocrop")
    mocker.patch.object(cog, "_process_image", side_effect=slow_process)

    # Create a mock attachment that saves quickly but processes slowly
    mock_attachment = mocker.Mock(spec=discord.Attachment)
    mock_attachment.save.return_value = None
    mock_attachment.content_type = "image/png"
    mock_attachment.filename = "test.png"

    success, error_msg = await cog._handle_crop(
        await dpytest.get_message().channel.send("test"),  # Get context
        "square",
        mock_attachment,
    )

    assert not success
    assert "Image processing timed out" in error_msg


@pytest.mark.asyncio
async def test_concurrent_processing(
    bot_with_autocrop_cog: DemocracyBot, test_image: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Test concurrent image processing.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        test_image: Test image fixture
        tmp_path: Temporary directory path
    """
    cog = bot_with_autocrop_cog.get_cog("Autocrop")

    # Create multiple output paths
    output_paths = [tmp_path / f"output_{i}.png" for i in range(3)]

    # Process multiple images concurrently
    tasks = [cog._process_image(str(test_image), (1, 1), str(output_path)) for output_path in output_paths]

    results = await asyncio.gather(*tasks)

    # Verify all processes completed successfully
    assert all(results)

    # Verify each output file exists and is a valid image
    from PIL import Image

    for output_path in output_paths:
        assert output_path.exists()
        with Image.open(output_path) as img:
            width, height = img.size
            assert width == height  # Should be square
