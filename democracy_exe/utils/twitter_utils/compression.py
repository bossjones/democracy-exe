"""Media compression utilities."""
from __future__ import annotations

import asyncio
import pathlib

from typing import Final, Optional

import structlog


logger = structlog.get_logger(__name__)

from democracy_exe.shell import ProcessException, ShellConsole, run_coroutine_subprocess


# Constants for Discord upload limits
MAX_DISCORD_SIZE: Final[int] = 8_388_608  # 8MB in bytes
MAX_DISCORD_NITRO_SIZE: Final[int] = 52_428_800  # 50MB in bytes

# Valid file extensions
VIDEO_EXTENSIONS: Final[set[str]] = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm", ".mpeg", ".3gp"}
AUDIO_EXTENSIONS: Final[set[str]] = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".wma"}

class CompressionError(Exception):
    """Base exception for compression-related errors."""


async def compress_media(
    file_path: str | pathlib.Path,
    max_size: int = MAX_DISCORD_SIZE,
    script_path: str | None = None
) -> str:
    """Compress media file to meet Discord size requirements.

    Args:
        file_path: Path to media file
        max_size: Maximum allowed file size in bytes
        script_path: Optional path to compression script

    Returns:
        Path to compressed file

    Raises:
        CompressionError: If compression fails
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise CompressionError(f"File not found: {file_path}")

    # Check if compression is needed
    if file_path.stat().st_size <= max_size:
        return str(file_path)

    # Find compression script
    if script_path is None:
        script_path = pathlib.Path(__file__).parent.parent.parent / "scripts" / "compress-discord.sh"

    if not pathlib.Path(script_path).exists():
        raise CompressionError(f"Compression script not found: {script_path}")

    try:
        # Make script executable
        script_path = pathlib.Path(script_path)
        script_path.chmod(0o755) # type: ignore

        # Run compression script using shell utility
        cmd = f"{script_path} {file_path}"
        working_dir = str(file_path.parent)

        ShellConsole.message(f"Compressing file: {file_path}")
        await run_coroutine_subprocess(cmd, "file://" + str(file_path), working_dir)

        # Find compressed output file
        output_dir = file_path.parent
        compressed_name = f"25MB_{file_path.stem}"

        # Check for both mp4 and mp3 extensions
        compressed_path = output_dir / f"{compressed_name}.mp4"
        if not compressed_path.exists():
            compressed_path = output_dir / f"{compressed_name}.mp3"

        if not compressed_path.exists():
            raise CompressionError("Compressed output file not found")

        logger.info(f"Successfully compressed {file_path} to {compressed_path}")
        return str(compressed_path)

    except ProcessException as e:
        logger.exception("Compression failed")
        raise CompressionError(f"Failed to compress {file_path}: {e!s}") from e

def is_compressible(file_path: str | pathlib.Path) -> bool:
    """Check if file can be compressed.

    Args:
        file_path: Path to check

    Returns:
        True if file extension is supported
    """
    suffix = pathlib.Path(file_path).suffix.lower()
    return suffix in VIDEO_EXTENSIONS or suffix in AUDIO_EXTENSIONS
