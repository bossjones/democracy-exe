#!/usr/bin/env python3

"""
Logging utilities for the SandboxAgent project.

This module provides functions and utilities for configuring and managing logging in the SandboxAgent project.
It includes functions for setting up the global logger, handling exceptions, and filtering log messages.

Functions:
    global_log_config(log_level: LOG_LEVEL, json: bool = False) -> None:
        Configure the global logger with the specified log level and format.

    get_logger(name: str = "democracy_exe") -> Logger:
        Get a logger instance with the specified name.

    _log_exception(exc: BaseException, dev_mode: bool = False) -> None:
        Log an exception with the appropriate level based on the dev_mode setting.

    _log_warning(exc: BaseException, dev_mode: bool = False) -> None:
        Log a warning with the appropriate level based on the dev_mode setting.

    filter_out_serialization_errors(record: dict[str, Any]) -> bool:
        Filter out log messages related to serialization errors.

    filter_out_modules(record: dict[str, Any]) -> bool:
        Filter out log messages from the standard logging module.

Constants:
    LOGURU_FILE_FORMAT: str
        The log format string for file logging.

    NEW_LOGGER_FORMAT: str
        The new log format string for console logging.

    LOG_LEVEL: Literal
        The available log levels.

    MASKED: str
        A constant string used to mask sensitive data in logs.

Classes:
    Pii(str):
        A custom string class that masks sensitive data in logs based on the log_pii setting.
"""
# pylint: disable=no-member
# pylint: disable=consider-using-tuple
# pylint: disable=eval-used,no-member
# pyright: ignore[reportOperatorIssue]
# pyright: ignore[reportOptionalIterable]
# SOURCE: https://betterstack.com/community/guides/logging/loguru/

# FIXME: https://github.com/sweepai/sweep/blob/7d93c612568b8febd4aaf3c75810794bc10c09ae/sweepai/utils/event_logger.py#L7
# FIXME: https://github.com/sweepai/sweep/blob/7d93c612568b8febd4aaf3c75810794bc10c09ae/sweepai/utils/event_logger.py#L7
# FIXME: https://github.com/sweepai/sweep/blob/7d93c612568b8febd4aaf3c75810794bc10c09ae/sweepai/utils/event_logger.py#L7
# FIXME: https://github.com/sweepai/sweep/blob/7d93c612568b8febd4aaf3c75810794bc10c09ae/sweepai/utils/event_logger.py#L7
# FIXME: https://github.com/sweepai/sweep/blob/7d93c612568b8febd4aaf3c75810794bc10c09ae/sweepai/utils/event_logger.py#L7
# FIXME: https://github.com/sweepai/sweep/blob/7d93c612568b8febd4aaf3c75810794bc10c09ae/sweepai/utils/event_logger.py#L7
# FIXME: https://github.com/sweepai/sweep/blob/7d93c612568b8febd4aaf3c75810794bc10c09ae/sweepai/utils/event_logger.py#L7
# FIXME: https://github.com/sweepai/sweep/blob/7d93c612568b8febd4aaf3c75810794bc10c09ae/sweepai/utils/event_logger.py#L7
# FIXME: https://github.com/sweepai/sweep/blob/7d93c612568b8febd4aaf3c75810794bc10c09ae/sweepai/utils/event_logger.py#L7

from __future__ import annotations

import contextvars
import functools
import gc
import inspect
import logging
import multiprocessing
import os
import re
import sys
import time

from datetime import UTC, datetime, timezone
from logging import Logger, LogRecord
from pathlib import Path
from pprint import pformat
from sys import stdout
from time import process_time
from types import FrameType
from typing import TYPE_CHECKING, Any, Deque, Dict, Literal, Optional, Union, cast

import loguru

from loguru import logger
from loguru._defaults import LOGURU_FORMAT
from loguru._logger import Core
from tqdm import tqdm

from democracy_exe.aio_settings import aiosettings
from democracy_exe.models.loggers import LoggerModel, LoggerPatch


if TYPE_CHECKING:
    from better_exceptions.log import BetExcLogger
    from loguru._logger import Logger as _Logger


LOGLEVEL_MAPPING = {
    50: "CRITICAL",
    40: "ERROR",
    30: "WARNING",
    20: "INFO",
    10: "DEBUG",
    5: "VVVV",
    0: "NOTSET",
}

LOGURU_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level}</level> | "
    "<cyan>{module}</cyan>:<cyan>{line}</cyan> | "
    "<level>{extra[room_id]}</level> - "
    "<level>{message}</level>"
)

LOGURU_FILE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{extra[room_id]}</level> - "
    "<level>{message}</level>"
)

# NOTE: this is the default format for loguru
_LOGURU_FORMAT = (
    "LOGURU_FORMAT",
    str,
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)

# NOTE: this is the new format for loguru
NEW_LOGGER_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<magenta>{file}:{line}</magenta> | "
    "<level>{message}</level> | {extra}"
)


LOG_LEVEL = Literal[
    "TRACE",
    "VVVV",
    "DEBUG",
    "INFO",
    "SUCCESS",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

# --------------------------------------------------------------------------
# SOURCE: https://github.com/ethyca/fides/blob/e9174b771155abc0ff3faf1fb7b8b76836788617/src/fides/api/util/logger.py#L116
MASKED = "MASKED"


class Pii(str):
    """Mask pii data"""

    def __format__(self, __format_spec: str) -> str:
        """
        Mask personally identifiable information (PII) in log messages.

        This class is a subclass of str and overrides the __format__ method to mask
        PII data in log messages. If the aiosettings.log_pii setting is True, the
        original string value is returned. Otherwise, the string "MASKED" is returned.

        Args:
            __format_spec (str): The format specification string.

        Returns:
            str: The masked or unmasked string value.
        """
        if aiosettings.log_pii:
            return super().__format__(__format_spec)
        return MASKED


def _log_exception(exc: BaseException, dev_mode: bool = False) -> None:
    """
    Log an exception with the appropriate level based on the dev_mode setting.

    If dev_mode is True, the entire traceback will be logged using logger.opt(exception=True).error().
    Otherwise, only the exception message will be logged using logger.error().

    Args:
        exc (BaseException): The exception to be logged.
        dev_mode (bool, optional): Whether to log the entire traceback or just the exception message.
            Defaults to False.

    Returns:
        None
    """
    if dev_mode:
        logger.opt(exception=True).error(exc)
    else:
        logger.error(exc)


def _log_warning(exc: BaseException, dev_mode: bool = False) -> None:
    """
    Log a warning with the appropriate level based on the dev_mode setting.

    If dev_mode is True, the entire traceback will be logged using logger.opt(exception=True).warning().
    Otherwise, only the warning message will be logged using logger.warning().

    Args:
        exc (BaseException): The exception or warning to be logged.
        dev_mode (bool, optional): Whether to log the entire traceback or just the warning message.
            Defaults to False.

    Returns:
        None
    """
    if dev_mode:
        logger.opt(exception=True).warning(exc)
    else:
        logger.warning(exc)


def filter_out_serialization_errors(record: dict[str, Any]):
    # Patterns to match the log messages you want to filter out
    patterns = [
        r"Orjson serialization failed:",
        r"Failed to serialize .* to JSON:",
        r"Object of type .* is not JSON serializable",
        # Failed to deepcopy input: TypeError("cannot pickle '_thread.RLock' object") | {}
        r".*Failed to deepcopy input:.*",
        r".*logging:callHandlers.*",
    ]

    # Check if the log message matches any of the patterns
    for pattern in patterns:
        if re.search(pattern, record["message"]):
            return False  # Filter out this message

    return True  # Keep all other messages


def filter_discord_logs(record: dict[str, Any]) -> bool:
    """
    Filter Discord.py log messages based on specific patterns and levels.

    This filter helps reduce noise from Discord.py's verbose logging while keeping
    important messages. It filters out common heartbeat messages and routine WebSocket
    events unless they indicate errors.

    Args:
        record: The log record to check.

    Returns:
        bool: True if the message should be kept, False if it should be filtered out.
    """
    # Always keep error and critical messages
    if record["level"].no >= 40:  # ERROR and CRITICAL
        return True

    # Filter out common Discord.py noise patterns
    noise_patterns = [
        r"^Shard ID \d* has sent the HEARTBEAT payload\.$",
        r"^Shard ID \d* has successfully IDENTIFIED\.$",
        r"^Shard ID \d* has sent the RESUME payload\.$",
        r"^Got a request to RESUME the session\.$",
        r"^WebSocket Event: \'PRESENCE_UPDATE\'\.$",
        r"^WebSocket Event: \'GUILD_CREATE\'\.$",
        r"^WebSocket Event: \'TYPING_START\'\.$",
        r"^Found matching ID for \w+ for \d+\.$",
    ]

    # Skip filtering if not from discord
    if not record["name"].startswith("discord"):
        return True

    # Check message against noise patterns
    message = str(record["message"])
    for pattern in noise_patterns:
        if re.match(pattern, message):
            return False

    return True

def filter_out_modules(record: dict[str, Any]) -> bool:
    """
    Filter out log messages from the standard logging module.

    Args:
        record: The log record to check.

    Returns:
        bool: True if the message should be kept, False if it should be filtered out.
    """
    # Check if the log message originates from the logging module
    if record["name"].startswith("logging"):
        return False  # Filter out this message

    return True  # Keep all other messages


def catch_all_filter(record: dict[str, Any]) -> bool:
    """
    Filter out log messages that match certain patterns or originate from specific modules.

    This function combines the `filter_out_serialization_errors` and `filter_out_modules` filters
    to create a single filter that removes log messages that match certain patterns (related to
    serialization errors) or originate from the standard logging module.

    Args:
        record (dict[str, Any]): The log record to check.

    Returns:
        bool: True if the message should be kept, False if it should be filtered out.
    """

    return filter_out_serialization_errors(record) and filter_out_modules(record)


class TqdmOutputStream:
    """
    A custom output stream that writes to sys.stderr and supports tqdm progress bars.

    This class provides a write method that writes strings to sys.stderr using tqdm.write,
    which allows tqdm progress bars to be displayed correctly. It also provides an isatty
    method that returns whether sys.stderr is a terminal.

    This class is useful when you want to use tqdm progress bars in your application while
    still being able to write to the standard error stream.
    """

    def write(self, string: str = "") -> None:
        """
        Write the given string to sys.stderr using tqdm.write.

        Args:
            string (str): The string to write.
        """
        tqdm.write(string, file=sys.stderr, end="")

    def isatty(self) -> bool:
        """
        Return whether sys.stderr is a terminal.

        Returns:
            bool: True if sys.stderr is a terminal, False otherwise.
        """
        return sys.stderr.isatty()


_console_handler_id: int | None = None
_file_handler_id: int | None = None

_old_log_dir: str | None = None
_old_console_log_level: LOG_LEVEL | None = None
_old_backup_count: int | None = None

REQUEST_ID_CONTEXTVAR = contextvars.ContextVar("request_id", default=None)

# initialize the context variable with a default value
REQUEST_ID_CONTEXTVAR.set("notset")


def set_log_extras(record: dict[str, Any]) -> None:
    """Set extra log fields in the log record.

    Args:
        record: The log record to modify.
    """
    record["extra"]["datetime"] = datetime.now(UTC)

# SOURCE: https://github.com/joint-online-judge/fastapi-rest-framework/blob/b0e93f0c0085597fcea4bb79606b653422f16700/fastapi_rest_framework/logging.py#L43
def format_record(record: dict[str, Any]) -> str:
    """Custom format for loguru loggers.

    Uses pformat for log any data like request/response body during debug.
    Works with logging if loguru handler it.

    Args:
        record: The log record.

    Returns:
        The formatted log record.
    """
    # Format the datetime in the expected format
    time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Build the basic message format
    format_string = (
        f"{time_str} | "
        f"{record['level']['name']:<8} | "
        f"{record['name']}:{record['function']} - "
        f"{record['file']}:{record['line']} | "
        f"{record['message']}"
    )

    # Add payload if present
    if record["extra"].get("payload") is not None:
        payload = pformat(record["extra"]["payload"], indent=4, compact=True, width=88)
        format_string += f"\n{payload}"

    # Add exception if present
    if record["exception"]:
        format_string += f"\n{record['exception']}"

    return format_string + "\n"

# SOURCE: https://github.com/joint-online-judge/fastapi-rest-framework/blob/b0e93f0c0085597fcea4bb79606b653422f16700/fastapi_rest_framework/logging.py#L43
def format_record_improved(record: dict[str, Any]) -> str:
    """Custom format for loguru loggers.

    Uses pformat for log any data like request/response body during debug.
    Works with logging if loguru handler it.

    Args:
        record: The log record.

    Returns:
        The formatted log record.
    """
    # Format the time using the record's time field
    time_str = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Safely format the message, handling dictionary content
    message = record["message"]
    if isinstance(message, str) and ('{' in message and '}' in message):
        try:
            # Check if message looks like a dict string but avoid eval
            if message.strip().startswith('{') and message.strip().endswith('}'):
                # Use pformat directly on the string representation
                message = pformat(message, indent=2, width=80)
            # Otherwise keep original message
        except Exception:
            # If formatting fails, keep original message
            pass

    # Format the basic message
    formatted = (
        f"{time_str} | "
        f"{record['level'].name:<8} | "
        f"{record['name']}:{record['function']} - "
        f"{record['file'].name}:{record['line']} | "
        f"{message}"
    )

    # Add payload if present
    if record["extra"].get("payload") is not None:
        payload = pformat(record["extra"]["payload"], indent=2, width=80)
        formatted += f"\n{payload}"

    # Add exception if present
    if record["exception"]:
        formatted += f"\n{record['exception']}"

    return formatted


class InterceptHandler(logging.Handler):
    """
    Intercept all logging calls (with standard logging) into our Loguru Sink
    See: https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
    """

    loglevel_mapping = {
        logging.CRITICAL: "CRITICAL",
        logging.ERROR: "ERROR",
        logging.FATAL: "FATAL",
        logging.WARNING: "WARNING",
        logging.INFO: "INFO",
        logging.DEBUG: "DEBUG",
        1: "DUMMY",
        0: "NOTSET",
    }

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        """
        Intercept all logging calls (with standard logging) into our Loguru Sink.

        This method is called by the standard logging library whenever a log message
        is emitted. It converts the standard logging record into a Loguru log message
        and logs it using the Loguru logger.

        Args:
            record (logging.LogRecord): The standard logging record to be logged.

        Returns:
            None
        """
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            # First try to get the level by name
            level = logger.level(record.levelname).name
        except ValueError:
            # If the level doesn't exist, register it
            logger.level(record.levelname, no=record.levelno)
            level = record.levelname

        # NOTE: Original question: I don't quite understand the frame and depth aspects, can you try explaining it using a practical example? imagine there is a logger for "discord.client" for example
        # -------------------------------------------------------------------------------------------------------------
        # # If we didn't track frames properly
        # logger.info("Connecting to Discord...")
        # # Would show: "File 'logging/__init__.py', line 123" in the log
        # # This is wrong! We want to know it came from client.py
        # Here's an example of how it would work:

        # # discord/client.py
        # import logging

        # logger = logging.getLogger("discord.client")

        # class Client:
        #     def connect(self):
        #         logger.info("Connecting to Discord...")  # This is where our log originates

        # # When this log message is created, it creates a stack of function calls (frames) like this:
        # # Frame 3 (Top): logging/__init__.py - internal logging code
        # # Frame 2: logging/__init__.py - more logging internals
        # # Frame 1: discord/client.py - our actual Client.connect() method
        # # Frame 0 (Bottom): The script that called Client.connect()

        # # Your application code
        # from discord import Client
        # import logging

        # # Set up logging
        # logging.basicConfig(handlers=[InterceptHandlerImproved()])

        # client = Client()
        # client.connect()

        # # What happens when client.connect() logs:
        # # 1. discord.client calls logger.info("Connecting to Discord...")
        # # 2. This goes through standard logging
        # # 3. InterceptHandlerImproved receives it
        # # 4. It walks back through the frames:
        # #    - Skips logging internals
        # #    - Finds discord/client.py
        # # 5. Finally outputs something like:
        # #    "2024-01-10 12:34:56 | INFO | discord.client:connect:45 | Connecting to Discord..."
        # #    With the correct file, function, and line number!
        # -------------------------------------------------------------------------------------------------------------


        # NOTE: Here is how it works:
        # Starts at depth 2, keeps going until it finds non-logging frame
        # Frame 3 (depth 2) -> logging/__init__.py (skip)
        # Frame 2 (depth 3) -> logging/__init__.py (skip)
        # Frame 1 (depth 4) -> discord/client.py (found it!)

        # INFO: Find Caller Frame
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            # DISABLED 12/10/2021 # frame = cast(FrameType, frame.f_back)
            depth += 1

        loguru.logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


class InterceptHandlerImproved(logging.Handler):
    """
    Intercept all logging calls (with standard logging) into our Loguru Sink
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Intercept and redirect log messages from the standard logging module to Loguru.

        Args:
            record (logging.LogRecord): The standard logging record to be logged.
        """
        # Map standard logging levels to Loguru levels
        try:
            level = record.levelname
        except ValueError:
            level = "INFO"  # Default to INFO if level mapping fails


        # NOTE: Here is how it works:
        # Starts at depth 0, more precisely tracks frames
        # Current frame (depth 0) -> InterceptHandler.emit
        # Frame 3 (depth 1) -> logging/__init__.py (skip)
        # Frame 2 (depth 2) -> logging/__init__.py (skip)
        # Frame 1 (depth 3) -> discord/client.py (found it!)

        # Find the stack frame from which the log message originated.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            # Traverse back through the stack frames.
            frame = frame.f_back
            # Increment the depth counter to track how far back we've gone.
            depth += 1

        # Log the message using Loguru, preserving the original context and exception info.
        logger.opt(depth=depth, exception=record.exc_info).log(
            level,  # Use the determined log level.
            record.getMessage(),  # Log the message content.
        )

class InterceptHandlerImproved2(logging.Handler):
    def __init__(self):
        super().__init__()
        self._is_logging = False

    def emit(self, record: logging.LogRecord) -> None:
        if self._is_logging:
            return  # Prevent recursive logging

        self._is_logging = True
        try:
            # Map standard logging levels to Loguru levels
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            # Find the caller's stack frame
            frame, depth = inspect.currentframe(), 0
            while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
                frame = frame.f_back
                depth += 1

            # Create a simplified message that won't trigger recursion
            message = str(record.getMessage())

            # Handle dictionary-like messages specially
            if message.startswith('{') and message.endswith('}'):
                try:
                    # Safely evaluate dictionary-like strings
                    message = f"Dict: {message}"
                except Exception:
                    pass

            logger.opt(depth=depth, exception=record.exc_info).log(
                level,
                message
            )
        finally:
            self._is_logging = False


class InterceptHandler3(logging.Handler):
    """
    An advanced logging handler that intercepts standard logging messages and redirects them to Loguru.
    This handler provides improved stack trace handling and level mapping.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Process and emit a logging record by converting it to a Loguru log message.

        Args:
            record: The logging record to process and emit.
        """
        # Try to map the standard logging level name to a Loguru level
        # For example, converts 'INFO' to Loguru's info level
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            # If the level name isn't recognized by Loguru, fall back to the numeric level
            level = record.levelno

        # Initialize frame tracking to find the true origin of the log message
        frame, depth = inspect.currentframe(), 0

        # Walk up the stack frames until we find the actual caller
        while frame:
            # Get the filename of the current frame
            filename = frame.f_code.co_filename

            # Check if this frame is from the logging module itself
            is_logging = filename == logging.__file__

            # Check if this frame is from Python's import machinery
            # These frames appear when code is running from a frozen executable
            is_frozen = "importlib" in filename and "_bootstrap" in filename

            # If we've gone past the logging frames and aren't in import machinery,
            # we've found our caller
            if depth > 0 and not (is_logging or is_frozen):
                break

            # Move up to the next frame and increment our depth counter
            frame = frame.f_back
            depth += 1

        # Emit the log message through Loguru with:
        # - proper stack depth for correct file/line reporting
        # - original exception info preserved
        # - original message content
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# logging.basicConfig(handlers=[InterceptHandler3()], level=0, force=True)

def get_logger(
    name: str,
    provider: str | None = None,
    level: int = logging.INFO,
    logger: logging.Logger = logger,
) -> logging.Logger:
    """
    Get a logger instance with the specified name and configuration.

    Args:
        name (str): The name of the logger.
        provider (Optional[str], optional): The provider for the logger. Defaults to None.
        level (int, optional): The logging level. Defaults to logging.INFO.
        logger (logging.Logger, optional): The logger instance to use. Defaults to the root logger.

    Returns:
        logging.Logger: The configured logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is an info message")
    """

    return logger


def request_id_filter(record: dict[str, Any]):
    """
    Inject the request id from the context var to the log record. The logging
    config format is defined in logger_config.yaml and has request_id as a field.
    """
    record["extra"]["request_id"] = REQUEST_ID_CONTEXTVAR.get()


def reset_logging(
    log_dir: str,
    *,
    console_log_level: LOG_LEVEL = "INFO",
    backup_count: int | None = None,
) -> None:
    """
    Reset the logging configuration.

    This function resets the logging configuration by removing any existing
    handlers and adding new handlers for console and file logging. The console
    log level and file backup count can be specified.

    Args:
        log_dir (str): The directory path for the log file.
        console_log_level (LOG_LEVEL, optional): The log level for the console
            handler. Defaults to "INFO".
        backup_count (Optional[int], optional): The number of backup log files
            to keep. If None, no backup files are kept. Defaults to None.

    Returns:
        None
    """
    global _console_handler_id, _file_handler_id
    global _old_log_dir, _old_console_log_level, _old_backup_count
    logger.configure(extra={"room_id": ""})

    if console_log_level != _old_console_log_level:
        if _console_handler_id is not None:
            logger.remove(_console_handler_id)
        else:
            logger.remove()  # remove the default stderr handler

        _console_handler_id = logger.add(
            sys.stderr,
            level=console_log_level,
            format=LOGURU_CONSOLE_FORMAT,
        )

        _old_console_log_level = console_log_level

    # Add file handler if log_dir is provided
    if log_dir:
        if _file_handler_id is not None:
            logger.remove(_file_handler_id)

        log_file = Path(log_dir) / "app.log"
        _file_handler_id = logger.add(
            log_file,
            rotation="1 MB",
            retention=backup_count,
            format=NEW_LOGGER_FORMAT,
            enqueue=True,
        )

        _old_log_dir = log_dir
        _old_backup_count = backup_count


def timeit(func):
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("Function '{}' executed in {:f} s", func.__name__, end - start)
        return result

    return wrapped


def logger_wraps(*, entry=True, exit=True, level="DEBUG"):
    def wrapper(func):
        name = func.__name__

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            logger_ = logger.opt(depth=1)
            if entry:
                logger_.log(level, "Entering '{}' (args={}, kwargs={})", name, args, kwargs)
            result = func(*args, **kwargs)
            if exit:
                logger_.log(level, "Exiting '{}' (result={})", name, result)
            return result

        return wrapped

    return wrapper


# @pysnooper.snoop()
# @pysnooper.snoop(thread_info=True)
# FIXME: https://github.com/abnerjacobsen/fastapi-mvc-loguru-demo/blob/main/mvc_demo/core/loguru_logs.py
# SOURCE: https://loguru.readthedocs.io/en/stable/api/logger.html#loguru._logger.Logger
def global_log_config(
    log_level: str | int = logging.INFO,
    json: bool = False,
    mp_context: str = "spawn"
) -> _Logger:
    """Configure global logging settings.

    Args:
        log_level: The log level to use. Defaults to logging.INFO.
        json: Whether to format logs as JSON. Defaults to False.
        mp_context: Multiprocessing context to use. One of: fork, spawn, forkserver.
                   Defaults to "spawn" for better cross-platform compatibility.

    Returns:
        The configured logger instance.
    """
    # Set up multiprocessing context
    context = multiprocessing.get_context(mp_context)

    # SOURCE: https://github.com/acgnhiki/blrec/blob/975fa2794a3843a883597acd5915a749a4e196c8/src/blrec/logging/configure_logging.py#L21
    global _console_handler_id, _file_handler_id
    global _old_log_dir, _old_console_log_level, _old_backup_count

    # If log_level is a string (e.g., "INFO", "DEBUG"), convert it to the corresponding numeric value
    # This allows users to specify log levels either as strings or as integer constants
    if isinstance(log_level, str):
        try:
            log_level = logging._nameToLevel.get(log_level.upper(), logging.INFO)
        except (AttributeError, KeyError):
            # If conversion fails, default to INFO level
            log_level = logging.INFO

    # NOTE: Original
    # intercept_handler = InterceptHandler()
    intercept_handler = InterceptHandlerImproved2()
    # intercept_handler = InterceptHandler3()

    log_filters = {
        "discord": aiosettings.thirdparty_lib_loglevel,
        # "telethon": aiosettings.thirdparty_lib_loglevel,
        # "web3": aiosettings.thirdparty_lib_loglevel,
        # "apprise": aiosettings.thirdparty_lib_loglevel,
        "urllib3": aiosettings.thirdparty_lib_loglevel,
        # "asyncz": aiosettings.thirdparty_lib_loglevel,
        # "rlp": aiosettings.thirdparty_lib_loglevel,
        # "numexpr": aiosettings.thirdparty_lib_loglevel,
        # "yfinance": aiosettings.thirdparty_lib_loglevel,
        # "peewee": aiosettings.thirdparty_lib_loglevel,
        "httpx": aiosettings.thirdparty_lib_loglevel,
        "openai": aiosettings.thirdparty_lib_loglevel,
        "httpcore": aiosettings.thirdparty_lib_loglevel,
    }

    logging.root.setLevel(log_level)

    # Create a set to track which module loggers we've already configured
    seen = set()
    # Iterate through a list of logger names, including:
    # 1. All existing loggers from the root logger's dictionary
    # 2. Specific modules we want to ensure are configured
    for name in [
        *logging.root.manager.loggerDict.keys(),  # pylint: disable=no-member
        "asyncio",          # Python's async/await framework
        "discord",          # Main Discord.py library
        "discord.client",   # Discord client module
        "discord.gateway",  # Discord WebSocket gateway
        "discord.http",     # Discord HTTP client
        "chromadb",        # Vector database for embeddings
        "langchain_chroma", # LangChain integration with ChromaDB
    ]:
        # Only process each base module once (e.g., 'discord.client' and 'discord.http'
        # both have base module 'discord')
        if name not in seen:
            if "discord" in name:
                print(f"name: {name}")
            # Extract the base module name (e.g., 'discord.client' -> 'discord')
            module_name = name.split(".")[0]
            # Log which module we're configuring
            print(f"Setting up logger for {module_name}")
            # Add the base module to our tracking set
            seen.add(module_name)
            # Replace the module's logging handlers with our custom interceptor
            module_logger = logging.getLogger(module_name)
            module_logger.handlers = [intercept_handler]

            if "discord" in name:
                module_logger.setLevel(logging.INFO)
            # logging.getLogger(name).handlers = [intercept_handler]

    # Get a new logger instance with multiprocessing support
    config = {
        "handlers": [{
            "sink": stdout,
            "format": format_record,
            # "filter": lambda record: (
            #     filter_out_serialization_errors(record)
            #     and filter_out_modules(record)
            #     and filter_discord_logs(record)
            # ),
            "filter": lambda record: (
                filter_out_serialization_errors(record)
                and filter_out_modules(record)
                and filter_discord_logs(record)
            ),
            "enqueue": True,  # Enable multiprocessing-safe queue
            "serialize": False,
            "backtrace": True,
            "diagnose": True,
            "catch": True,
            "level": log_level,
            "colorize": True,
            # "flush": True,
        }],
        "extra": {"request_id": REQUEST_ID_CONTEXTVAR.get()}
    }

    new_logger = logger.configure(**config)

    # Register cleanup handlers for multiprocessing
    import atexit
    import signal

    def cleanup_logger():
        """Ensure all logging is complete at exit."""
        try:
            logger.complete()
        except Exception:
            pass

    def terminate_handler(signo, frame):
        """Handle termination gracefully."""
        cleanup_logger()
        sys.exit(signo)

    atexit.register(cleanup_logger)
    signal.signal(signal.SIGTERM, terminate_handler)
    signal.signal(signal.SIGINT, terminate_handler)

    print(f"Logger set up with log level: {log_level}")

    setup_uvicorn_logger()
    setup_gunicorn_logger()

    return logger


def setup_uvicorn_logger():
    """
    Set up the uvicorn logger.

    This function configures the uvicorn logger to use the InterceptHandler,
    which allows for better handling and formatting of log messages from uvicorn.
    """
    loggers = (logging.getLogger(name) for name in logging.root.manager.loggerDict if name.startswith("uvicorn."))
    for uvicorn_logger in loggers:
        uvicorn_logger.handlers = []
    logging.getLogger("uvicorn").handlers = [InterceptHandler()]


def setup_gunicorn_logger():
    logging.getLogger("gunicorn.error").handlers = [InterceptHandler()]
    logging.getLogger("gunicorn.access").handlers = [InterceptHandler()]

def setup_discord_logger():
    discord_logger = logging.getLogger('discord')
    discord_logger.setLevel(logging.INFO)
    discord_logger.addHandler(InterceptHandler())


def get_lm_from_tree(loggertree: LoggerModel, find_me: str) -> LoggerModel | None:
    """Recursively search for a logger model in the logger tree.

    Args:
        loggertree: The root logger model to search from.
        find_me: The name of the logger model to find.

    Returns:
        The found logger model, or None if not found.
    """
    if find_me == loggertree.name:
        print("Found")
        return loggertree
    else:
        for ch in loggertree.children:
            print(f"Looking in: {ch.name}")
            if i := get_lm_from_tree(ch, find_me):
                return i
    return None


def generate_tree() -> LoggerModel:
    """Generate a tree of logger models.

    Returns:
        The root logger model of the generated tree.
    """
    rootm = LoggerModel(name="root", level=logging.getLogger().getEffectiveLevel(), children=[])
    nodesm: dict[str, LoggerModel] = {}
    items = sorted(logging.root.manager.loggerDict.items())  # type: ignore
    for name, loggeritem in items:
        if isinstance(loggeritem, logging.PlaceHolder):
            nodesm[name] = nodem = LoggerModel(name=name, children=[])
        else:
            nodesm[name] = nodem = LoggerModel(name=name, level=loggeritem.getEffectiveLevel(), children=[])
        i = name.rfind(".", 0, len(name) - 1)
        parentm = rootm if i == -1 else nodesm[name[:i]]
        parentm.children.append(nodem)
    return rootm


def obfuscate_message(message: str) -> str:
    """Obfuscate sensitive information in a message.

    Args:
        message: The message to obfuscate.

    Returns:
        The obfuscated message.
    """
    obfuscation_patterns = [
        (r"email: .*", "email: ******"),
        (r"password: .*", "password: ******"),
        (r"newPassword: .*", "newPassword: ******"),
        (r"resetToken: .*", "resetToken: ******"),
        (r"authToken: .*", "authToken: ******"),
        (r"located at .*", "located at ******"),
        (r"#token=.*", "#token=******"),
    ]
    for pattern, replacement in obfuscation_patterns:
        message = re.sub(pattern, replacement, message)

    return message


def formatter(record: dict[str, Any]) -> str:
    """Format a log record.

    Args:
        record: The log record to format.

    Returns:
        The formatted log record.
    """
    record["extra"]["obfuscated_message"] = record["message"]
    return (
        "<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> <level>[{level}]</level> - "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{extra[obfuscated_message]}</level>\n{exception}"
    )


def formatter_sensitive(record: dict[str, Any]) -> str:
    """Format a log record with sensitive information obfuscated.

    Args:
        record: The log record to format.

    Returns:
        The formatted log record with sensitive information obfuscated.
    """
    record["extra"]["obfuscated_message"] = obfuscate_message(record["message"])
    return (
        "<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> <level>[{level}]</level> - "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{extra[obfuscated_message]}</level>\n{exception}"
    )


# SMOKE-TESTS
if __name__ == "__main__":
    from logging_tree import printout

    global_log_config(
        log_level=logging.getLevelName("DEBUG"),
        json=False,
    )
    LOGGER = logger

    def dump_logger_tree():
        rootm = generate_tree()
        LOGGER.debug(rootm)

    def dump_logger(logger_name: str):
        LOGGER.debug(f"getting logger {logger_name}")
        rootm = generate_tree()
        return get_lm_from_tree(rootm, logger_name)

    LOGGER.info("TESTING TESTING 1-2-3")
    printout()
