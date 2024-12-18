"""Shell utility functions for executing commands and managing processes."""
from __future__ import annotations

import asyncio
import functools

from collections.abc import Callable
from typing import Any, TypeVar

from loguru import logger

from democracy_exe.shell import _aio_run_process_and_communicate, run_coroutine_subprocess


T = TypeVar("T")


def to_async(func: Callable[..., T]) -> Callable[..., asyncio.Future[T]]:
    """Convert a synchronous function to an asynchronous one.

    Args:
        func: The synchronous function to convert

    Returns:
        An asynchronous version of the function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> asyncio.Future[T]:
        return asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))
    return wrapper


# async def _aio_run_process_and_communicate(
#     cmd: list[str], cwd: str | None = None
# ) -> str:
#     """Run a process asynchronously and get its output.

#     Args:
#         cmd: Command to run as list of strings
#         cwd: Working directory for the command

#     Returns:
#         The command output as a string

#     Raises:
#         subprocess.CalledProcessError: If command execution fails
#     """
#     try:
#         process = await asyncio.create_subprocess_exec(
#             *cmd,
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE,
#             cwd=cwd
#         )
#         stdout, stderr = await process.communicate()

#         if process.returncode != 0:
#             logger.error(f"Command failed: {' '.join(cmd)}")
#             logger.error(f"stderr: {stderr.decode()}")
#             raise RuntimeError(f"Command failed with return code {process.returncode}")

#         return stdout.decode().strip()
#     except Exception as e:
#         logger.error(f"Error running command {cmd}: {e}")
#         raise


# async def run_coroutine_subprocess(cmd: str, uri: str) -> None:
#     """Run a subprocess coroutine.

#     Args:
#         cmd: Command to run
#         uri: URI for the command

#     Raises:
#         RuntimeError: If command execution fails
#     """
#     try:
#         process = await asyncio.create_subprocess_shell(
#             cmd,
#             stdout=asyncio.subprocess.PIPE,
#             stderr=asyncio.subprocess.PIPE
#         )
#         stdout, stderr = await process.communicate()

#         if process.returncode != 0:
#             logger.error(f"Command failed: {cmd}")
#             logger.error(f"stderr: {stderr.decode()}")
#             raise RuntimeError(f"Command failed with return code {process.returncode}")

#         logger.info(f"Command output: {stdout.decode()}")
#     except Exception as e:
#         logger.error(f"Error running command {cmd}: {e}")
#         raise


async_ = to_async  # Alias for backward compatibility
