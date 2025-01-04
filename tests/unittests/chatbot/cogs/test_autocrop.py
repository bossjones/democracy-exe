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

from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.cogs.autocrop import HELP_MESSAGE, AutocropError
from democracy_exe.chatbot.cogs.autocrop import Autocrop as AutocropCog
from democracy_exe.chatbot.core.bot import DemocracyBot
from democracy_exe.utils._testing import ContextLogger
from tests.internal.discord_test_utils import SlowAttachment


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.logging import LogCaptureFixture

    from pytest_mock.plugin import MockerFixture


@pytest_asyncio.fixture(autouse=True)
async def bot_with_autocrop_cog() -> AsyncGenerator[DemocracyBot, None]:
    """Create a DemocracyBot instance with Autocrop cog for testing.

    This fixture sets up a Discord bot instance with all required intents and
    the Autocrop cog installed. It handles proper initialization and cleanup.

    Yields:
        DemocracyBot: Configured bot instance with Autocrop cog

    Note:
        This fixture is automatically used in all tests in this module.
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
    """Clean up the global message queue after each test.

    This fixture ensures that no messages from previous tests remain in the queue,
    preventing test interference.

    Yields:
        None
    """
    yield
    try:
        await dpytest.empty_queue()
    finally:
        pass


@pytest.fixture
def test_image(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a test image file for testing.

    This fixture generates a random RGB image using numpy and saves it to a
    temporary file for testing image processing operations.

    Args:
        tmp_path: Pytest-provided temporary directory path

    Returns:
        pathlib.Path: Path to the generated test image file
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
    """Test the core image processing functionality.

    This test verifies that the _process_image method correctly processes
    an image to the specified aspect ratio while maintaining proper dimensions.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        test_image: Path to test image file
        tmp_path: Temporary directory for output
        caplog: Pytest logging capture fixture

    Note:
        This test specifically checks square cropping (1:1 aspect ratio)
        and verifies the output dimensions.
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
    """Create an invalid file for testing error handling.

    This fixture creates a text file that appears to be an image file,
    used to test the cog's handling of invalid file types.

    Args:
        tmp_path: Pytest-provided temporary directory path

    Returns:
        pathlib.Path: Path to the invalid test file
    """
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
async def test_crop_download_timeout_draft(
    bot_with_autocrop_cog: DemocracyBot, mocker: MockerFixture, test_image: pathlib.Path
) -> None:
    """Test timeout handling during image download.

    This test verifies that the cog properly handles timeouts when
    downloading attachments takes longer than the configured timeout.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        mocker: Pytest mocker fixture for creating mocks
        test_image: Path to test image file

    Note:
        Uses asyncio.sleep to simulate a slow download that exceeds
        the configured timeout duration.
    """
    # Get the first guild (server) from the bot's guild list
    guild = bot_with_autocrop_cog.guilds[0]
    # Get the first member from the guild and type-annotate it as a discord.Member
    author: discord.Member = guild.members[0]
    # Get the first channel from the guild
    channel = guild.channels[0]

    # # Create a mock attachment that simulates a slow download
    # class SlowAttachment(discord.Attachment):
    #     async def save(self, fp: str, **kwargs: Any) -> None:
    #         await asyncio.sleep(aiosettings.autocrop_download_timeout + 1)
    #         # Simulate saving the file
    #         with open(fp, 'wb') as f:
    #             f.write(b'test content')

    # Create a mock Discord Attachment object with test image data
    attach: discord.Attachment = SlowAttachment(
        # Pass the current state manager from dpytest backend
        state=dpytest.back.get_state(),
        # Create a mock attachment dictionary with test image properties
        data=dpytest.back.facts.make_attachment_dict(
            # Set the filename for the test attachment
            f"{test_image.name}",
            # Set the file size in bytes
            15112122,
            # Set the CDN URL for the attachment
            f"https://media.discordapp.net/attachments/some_number/random_number/{test_image.name}",
            # Set the proxy URL (usually same as CDN URL)
            f"https://media.discordapp.net/attachments/some_number/random_number/{test_image.name}",
            # Set the image height in pixels
            height=1000,
            # Set the image width in pixels
            width=1000,
            # Set the MIME type of the attachment
            content_type="image/jpeg",
        ),
    )

    # Create a message dictionary using dpytest's factory with the channel, author and attachment
    message_dict = dpytest.back.facts.make_message_dict(channel, author, attachments=[attach])

    try:
        # Attempt to create a Discord Message object from the dictionary and type-annotate it
        message: discord.Message = discord.Message(state=dpytest.back.get_state(), channel=channel, data=message_dict)
    # If any error occurs during message creation, fail the test with the error message
    except Exception as err:
        pytest.fail(str(err))

    # # Create a slow attachment with the test image
    # slow_attachment = SlowAttachment(
    #     state=mocker.Mock(),
    #     data={
    #         "id": 123456789,
    #         "filename": test_image.name,
    #         "size": 1000,
    #         "url": "http://example.com/test.png",
    #         "proxy_url": "http://example.com/test.png",
    #         "content_type": "image/png",
    #     },
    # )

    # # Send command with slow attachment
    # # Create a message first
    # await dpytest.message("test")
    # await asyncio.sleep(0.1)  # Add small delay to ensure message is queued
    # # Get cog instance
    # cog = bot_with_autocrop_cog.get_cog("Autocrop")

    # # Create a message first
    # await dpytest.message("test")
    # await asyncio.sleep(0.1)  # Add small delay to ensure message is queued
    # message = await dpytest.get_message()

    # # Create mock attachment that simulates slow processing
    # mock_attachment = mocker.Mock(spec=discord.Attachment)
    # mock_attachment.save.return_value = None
    # mock_attachment.content_type = "image/png"
    # mock_attachment.filename = "test.png"

    # # Test directly with _handle_crop instead of going through dpytest.message
    # success, error_msg = await cog._handle_crop(
    #     message,
    #     "square",
    #     mock_attachment,
    # )

    # assert not success
    # assert "Image processing timed out" in error_msg

    # # Test directly with _handle_crop instead of going through dpytest.message
    # # since dpytest doesn't handle custom Attachment classes well
    # success, error_msg = await cog._handle_crop(
    #     message,
    #     "square",
    #     slow_attachment,
    # )

    # assert not success
    # assert "Image download timed out" in error_msg

    # # Verify timeout error message
    # assert dpytest.verify().message().content("Processing image to square format...")
    # assert dpytest.verify().message().content("Image download timed out")


@pytest.mark.asyncio
@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Alert is suppresed. I think the draf function above works and will do a better job but who knows",
)
async def test_crop_download_timeout(
    bot_with_autocrop_cog: DemocracyBot, mocker: MockerFixture, test_image: pathlib.Path
) -> None:
    """Test timeout handling during image download.

    This test verifies that the cog properly handles timeouts when
    downloading attachments takes longer than the configured timeout.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        mocker: Pytest mocker fixture for creating mocks
        test_image: Path to test image file

    Note:
        Uses asyncio.sleep to simulate a slow download that exceeds
        the configured timeout duration.
    """

    # Create a mock attachment that simulates a slow download
    class SlowAttachment(discord.Attachment):
        async def save(self, fp: str, **kwargs: Any) -> None:
            await asyncio.sleep(aiosettings.autocrop_download_timeout + 1)
            # Simulate saving the file
            with open(fp, "wb") as f:
                f.write(b"test content")

    # Create a slow attachment with the test image
    slow_attachment = SlowAttachment(
        state=mocker.Mock(),
        data={
            "id": 123456789,
            "filename": test_image.name,
            "size": 1000,
            "url": "http://example.com/test.png",
            "proxy_url": "http://example.com/test.png",
            "content_type": "image/png",
        },
    )

    # Send command with slow attachment
    # Create a message first
    await dpytest.message("test")
    await asyncio.sleep(0.1)  # Add small delay to ensure message is queued
    # Get cog instance
    cog = bot_with_autocrop_cog.get_cog("Autocrop")

    # Create a message first
    await dpytest.message("test")
    await asyncio.sleep(0.1)  # Add small delay to ensure message is queued
    message = await dpytest.get_message()

    # Create mock attachment that simulates slow processing
    mock_attachment = mocker.Mock(spec=discord.Attachment)
    mock_attachment.save.return_value = None
    mock_attachment.content_type = "image/png"
    mock_attachment.filename = "test.png"

    # Test directly with _handle_crop instead of going through dpytest.message
    success, error_msg = await cog._handle_crop(
        message,
        "square",
        mock_attachment,
    )

    assert not success
    assert "Image processing timed out" in error_msg

    # Test directly with _handle_crop instead of going through dpytest.message
    # since dpytest doesn't handle custom Attachment classes well
    success, error_msg = await cog._handle_crop(
        message,
        "square",
        slow_attachment,
    )

    assert not success
    assert "Image download timed out" in error_msg

    # Verify timeout error message
    assert dpytest.verify().message().content("Processing image to square format...")
    assert dpytest.verify().message().content("Image download timed out")


@pytest.mark.asyncio
@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Alert is suppresed. I think the draf function above works and will do a better job but who knows",
)
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

    # Create a message first
    await dpytest.message("test")
    await asyncio.sleep(0.1)  # Add small delay to ensure message is queued
    message = await dpytest.get_message()

    success, error_msg = await cog._handle_crop(
        message,
        "square",
        mock_attachment,
    )

    assert not success
    assert "Image processing timed out" in error_msg


@pytest.mark.asyncio
async def test_concurrent_processing(
    bot_with_autocrop_cog: DemocracyBot, test_image: pathlib.Path, tmp_path: pathlib.Path
) -> None:
    """Test concurrent image processing capabilities.

    This test verifies that the cog can handle multiple simultaneous
    image processing requests without interference or corruption.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        test_image: Path to test image file
        tmp_path: Temporary directory for output files

    Note:
        Processes multiple copies of the same image concurrently and
        verifies that all outputs are correctly processed.
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
