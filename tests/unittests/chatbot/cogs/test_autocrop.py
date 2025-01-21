# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
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
import discord.ext.commands as commands
import discord.ext.test as dpytest
import pytest_asyncio
import structlog

from structlog.testing import capture_logs

import pytest

from democracy_exe.aio_settings import aiosettings


logger = structlog.get_logger(__name__)
from democracy_exe.chatbot.cogs.autocrop import HELP_MESSAGE, AutocropError
from democracy_exe.chatbot.cogs.autocrop import Autocrop as AutocropCog
from democracy_exe.chatbot.core.bot import DemocracyBot
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

    # try:
    #     await dpytest.empty_queue()
    #     # Close bot's session and cleanup
    #     if test_bot.ws is not None:
    #         test_bot.ws.socket = None
    #         test_bot.ws.thread = None
    #     await test_bot.close()
    #     # Cancel all tasks
    #     for task in asyncio.all_tasks():
    #         if task is not asyncio.current_task():
    #             task.cancel()
    #     # Wait for tasks to complete
    #     await asyncio.sleep(0.1)
    # finally:
    #     # Final cleanup
    #     await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)


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


# @pytest.fixture(autouse=True)
# def configure_structlog() -> None:
#     """Configure structlog for testing.

#     This fixture ensures each test has a clean, properly configured structlog setup.
#     It disables caching and configures appropriate processors for testing.
#     """
#     structlog.reset_defaults()
#     structlog.configure(
#         processors=[
#             structlog.processors.TimeStamper(fmt="iso"),
#             structlog.processors.add_log_level,
#             structlog.processors.StackInfoRenderer(),
#             structlog.processors.format_exc_info,
#             structlog.testing.LogCapture(),
#         ],
#         wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
#         logger_factory=structlog.PrintLoggerFactory,
#         context_class=dict,
#         cache_logger_on_first_use=False,  # Important: Disable caching for tests
#     )


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
    with structlog.testing.capture_logs() as captured:
        cog = bot_with_autocrop_cog.get_cog("Autocrop")
        await cog.on_ready()

        print("\nCaptured logs in test_autocrop_cog_on_ready:")
        for log in captured:
            print(f"Log event: {log}")

        # Check if the log message exists in the captured structlog events
        assert any(log.get("event") == "Autocrop Cog ready." for log in captured), (
            "Expected 'Autocrop Cog ready.' message not found in logs"
        )


@pytest.mark.asyncio
async def test_autocrop_cog_on_guild_join(bot_with_autocrop_cog: DemocracyBot, caplog: LogCaptureFixture) -> None:
    """Test Autocrop cog on_guild_join event.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
        caplog: Pytest log capture fixture
    """
    with capture_logs() as captured:
        cog = bot_with_autocrop_cog.get_cog("Autocrop")
        guild = dpytest.get_config().guilds[0]
        await cog.on_guild_join(guild)

        print("\nCaptured logs in test_autocrop_cog_on_guild_join:")
        for log in captured:
            print(f"Log event: {log}")

        # Check if the log message exists in the captured structlog events
        assert any(log.get("event") == f"Adding new guild to database: {guild.id}" for log in captured), (
            "Expected 'Adding new guild to database' message not found in logs"
        )


@pytest.mark.asyncio
async def test_crop_help_command(bot_with_autocrop_cog: DemocracyBot) -> None:
    """Test crop help command shows help message.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
    """
    await dpytest.message("?crop")
    # Wait for message to be processed
    message = await dpytest.sent_queue.get()
    assert "Available commands" in message.content


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
    with capture_logs() as captured:
        cog = bot_with_autocrop_cog.get_cog("Autocrop")
        output_path = tmp_path / "output.png"

        # Create a test image with known dimensions
        import numpy as np

        from PIL import Image

        # Create a 200x100 test image (landscape)
        img_array = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        test_img = Image.fromarray(img_array)
        test_img_path = tmp_path / "test_landscape.png"
        test_img.save(test_img_path)

        # Test square crop (1:1)
        success = await cog._process_image(str(test_img_path), (1, 1), str(output_path))
        assert success, "Image processing should succeed"

        # Verify output image dimensions
        with Image.open(output_path) as img:
            width, height = img.size
            assert width == height, f"Output image should be square, got {width}x{height}"
            # Since input was 200x100, the square crop should be 100x100
            assert width == 100, f"Expected width 100, got {width}"
            assert height == 100, f"Expected height 100, got {height}"

        # Debug: Print captured logs
        print("\nCaptured logs in test_process_image:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event").startswith("Starting image processing") for log in captured), (
            "Expected 'Starting image processing' message not found in logs"
        )

        assert any(log.get("event") == "Image processed successfully" for log in captured), (
            "Expected 'Image processed successfully' message not found in logs"
        )


@pytest.mark.asyncio
async def test_crop_invalid_attachment(bot_with_autocrop_cog: DemocracyBot) -> None:
    """Test crop command with invalid attachment.

    Args:
        bot_with_autocrop_cog: The Discord bot instance with Autocrop cog
    """
    await dpytest.message("?crop square")
    # Wait for message to be processed
    message = await dpytest.sent_queue.get()
    assert "Please attach an image to crop" in message.content


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
    # Wait for message to be processed
    message = await dpytest.sent_queue.get()
    assert "Please provide a valid image file" in message.content


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
        Uses SlowAttachment to simulate a slow download that exceeds
        the configured timeout duration.
    """
    with capture_logs() as captured:
        # Get the first guild (server) from the bot's guild list
        guild = dpytest.get_config().guilds[0]
        # Get the first member from the guild
        author = guild.members[0]
        # Get the first channel from the guild
        channel = guild.channels[0]

        # Create a mock Discord Attachment object with test image data
        attach = SlowAttachment(
            state=dpytest.back.get_state(),
            data=dpytest.back.facts.make_attachment_dict(
                filename=test_image.name,
                size=15112122,
                url=f"https://media.discordapp.net/attachments/123/456/{test_image.name}",
                proxy_url=f"https://media.discordapp.net/attachments/123/456/{test_image.name}",
                height=1000,
                width=1000,
                content_type="image/png",
            ),
        )

        # Create a message dictionary using dpytest's factory
        message_dict = dpytest.back.facts.make_message_dict(channel, author, attachments=[attach])

        # Create a Discord Message object from the dictionary
        message = discord.Message(state=dpytest.back.get_state(), channel=channel, data=message_dict)

        # Get the Autocrop cog
        cog = bot_with_autocrop_cog.get_cog("Autocrop")

        # Create a mock context
        ctx = mocker.Mock(spec=commands.Context)
        ctx.message = message
        ctx.author = author
        ctx.guild = guild
        ctx.channel = channel
        ctx.send = mocker.AsyncMock()

        # Call _handle_crop directly with the message and attachment
        success, error_msg = await cog._handle_crop(ctx, "square", attach)

        # Verify operation failed as expected
        assert not success, "Expected crop operation to fail due to timeout"
        assert error_msg == "Image download timed out", f"Expected timeout error message, got: {error_msg}"

        # Debug: Print captured logs
        print("\nCaptured logs:")
        for log in captured:
            print(f"Log entry: {log}")

        # Verify logging - check exact event names and fields
        start_log = f"Starting crop operation - User: {ctx.author}, Guild: {ctx.guild}, Ratio: square"
        assert any(log.get("event") == start_log and log.get("log_level") == "info" for log in captured), (
            f"Expected '{start_log}' event not found in logs"
        )

        download_log = f"Downloading attachment with {aiosettings.autocrop_download_timeout}s timeout"
        assert any(log.get("event") == download_log and log.get("log_level") == "debug" for log in captured), (
            f"Expected '{download_log}' event not found in logs"
        )

        timeout_log = "Image download timed out"
        assert any(log.get("event") == timeout_log and log.get("log_level") == "error" for log in captured), (
            f"Expected '{timeout_log}' error event not found in logs"
        )

        # Verify proper error logging
        error_logs = [log for log in captured if log.get("log_level") == "error" and log.get("event") == timeout_log]
        assert len(error_logs) == 1, "Expected exactly one error log for timeout"


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
    with capture_logs() as captured:
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

        # Debug: Print captured logs
        print("\nCaptured logs in test_crop_processing_timeout:")
        for log in captured:
            print(f"Log event: {log}")

        # Verify logging using structlog's capture_logs
        assert any(log.get("event").startswith("Starting image processing") for log in captured), (
            "Expected 'Starting image processing' message not found in logs"
        )

        assert any(log.get("event") == "Image processing timed out" for log in captured), (
            "Expected 'Image processing timed out' message not found in logs"
        )


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
    with capture_logs() as captured:
        cog = bot_with_autocrop_cog.get_cog("Autocrop")

        # Create test images with different dimensions
        import numpy as np

        from PIL import Image

        # Create test images: landscape, portrait, and square
        test_images = []
        dimensions = [(200, 100), (100, 200), (150, 150)]

        for i, (width, height) in enumerate(dimensions):
            img_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            test_img = Image.fromarray(img_array)
            test_img_path = tmp_path / f"test_image_{i}.png"
            test_img.save(test_img_path)
            test_images.append(test_img_path)

        # Create multiple output paths
        output_paths = [tmp_path / f"output_{i}.png" for i in range(len(test_images))]

        # Process multiple images concurrently with different aspect ratios
        tasks = [
            cog._process_image(str(img_path), ratio, str(out_path))
            for img_path, out_path, ratio in zip(
                test_images,
                output_paths,
                [(1, 1), (4, 5), (16, 9)],
                strict=False,  # square, portrait, landscape
            )
        ]

        # Run tasks concurrently
        results = await asyncio.gather(*tasks)

        # Verify all processes completed successfully
        assert all(results), "All image processing tasks should succeed"

        # Verify each output file exists and has correct dimensions
        expected_ratios = [(1, 1), (4, 5), (16, 9)]
        for output_path, (target_width, target_height) in zip(output_paths, expected_ratios, strict=False):
            assert output_path.exists(), f"Output file {output_path} should exist"

            with Image.open(output_path) as img:
                width, height = img.size
                actual_ratio = width / height
                expected_ratio = target_width / target_height
                assert abs(actual_ratio - expected_ratio) < 0.01, (
                    f"Expected aspect ratio {expected_ratio:.2f}, got {actual_ratio:.2f}"
                )

        # Verify logging
        processing_starts = [log for log in captured if log.get("event", "").startswith("Starting image processing")]
        assert len(processing_starts) == len(test_images), (
            f"Expected {len(test_images)} 'Starting image processing' messages, got {len(processing_starts)}"
        )

        success_logs = [log for log in captured if "Image processed successfully" in str(log.get("event", ""))]
        assert len(success_logs) == len(test_images), (
            f"Expected {len(test_images)} 'Image processed successfully' messages, got {len(success_logs)}"
        )
