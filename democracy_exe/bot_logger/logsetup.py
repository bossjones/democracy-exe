from __future__ import annotations

import logging
import logging.config
import multiprocessing
import sys

from typing import Any, Dict, List, Optional, Union

import structlog

from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars, reset_contextvars
from structlog.stdlib import BoundLogger
from structlog.typing import Processor

from democracy_exe.aio_settings import aiosettings


# so we have logger names
structlog.stdlib.recreate_defaults()

logger: structlog.stdlib.BoundLogger  = structlog.get_logger(__name__)


def get_log_level(level: str | int) -> int:
    """Convert a log level string to its corresponding logging level value.

    This function takes a string or integer log level and returns the corresponding
    logging level value. It supports both string names (e.g., 'INFO', 'DEBUG') and
    integer values.

    Args:
        level: The log level as a string (e.g., 'INFO', 'DEBUG') or integer

    Returns:
        int: The corresponding logging level value

    Raises:
        ValueError: If the log level string is not valid
        TypeError: If the level is not a string or integer
    """
    if isinstance(level, int):
        if level < 0:
            raise ValueError(f"Invalid level value, it should be a positive integer, not: {level}")
        return level

    if not isinstance(level, str):
        raise TypeError(
            f"Invalid level, it should be an integer or a string, not: '{type(level).__name__}'"
        )

    # Convert to upper case for case-insensitive comparison
    level_upper = level.upper()

    # Map of level names to level numbers
    level_map = {
        'CRITICAL': logging.CRITICAL,  # 50
        'FATAL': logging.FATAL,        # 50
        'ERROR': logging.ERROR,        # 40
        'WARNING': logging.WARNING,    # 30
        'WARN': logging.WARN,          # 30
        'INFO': logging.INFO,          # 20
        'DEBUG': logging.DEBUG,        # 10
        'NOTSET': logging.NOTSET      # 0
    }

    try:
        return level_map[level_upper]
    except KeyError:
        raise ValueError(f"Invalid log level: '{level}'") from None


def _add_module_name(logger: str, method_name: str, event_dict: dict) -> dict:
    """Add the module name to the event dict.

    Args:
        logger: The logger name
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        dict: The event dictionary with module name added
    """
    event_dict["module"] = logger
    return event_dict


def _add_process_info(logger: str, method_name: str, event_dict: dict) -> dict:
    """Add process information to the event dict.

    Args:
        logger: The logger name
        method_name: The logging method name
        event_dict: The event dictionary

    Returns:
        dict: The event dictionary with process info added
    """
    try:
        process = multiprocessing.current_process()
        event_dict.update({
            "process_name": process.name,
            "process_id": process.pid,
        })
    except Exception:
        # Fail silently if we can't get process info
        pass
    return event_dict


def configure_logging(
    enable_json_logs: bool = False,
    log_level: str = "INFO",
    third_party_loggers: dict[str, str] | None = None
) -> dict[str, Any]:
    """Configure structured logging with comprehensive async support and third-party integration.

    This function sets up a unified logging system that:
    1. Configures both structlog and standard logging
    2. Provides async-safe logging using contextvars
    3. Properly captures third-party library logs
    4. Includes detailed context information
    5. Supports both JSON and console output
    6. Optimized for performance with caching
    7. Supports async context isolation
    8. Handles multiprocessing properly

    Args:
        enable_json_logs: If True, outputs logs in JSON format. Otherwise, uses colored console output.
        log_level: The minimum log level to capture.
        third_party_loggers: Dict of logger names and their minimum levels to configure.
    """
    # Default third-party logger configuration
    default_third_party = {
        "discord": "WARNING",
        "discord.client": "WARNING",
        "discord.gateway": "WARNING",
        "aiohttp": "WARNING",
        "asyncio": "WARNING",
        "urllib3": "WARNING",
        "requests": "WARNING",
        "PIL": "WARNING",
    }

    if third_party_loggers:
        default_third_party.update(third_party_loggers)

    # Clear any existing context
    clear_contextvars()


# borrowed this from /Users/malcolm/dev/structlog/tests/typing/api.py
#     structlog.configure(
#     processors=[
#         structlog.stdlib.filter_by_level,
#         structlog.stdlib.add_logger_name,
#         structlog.stdlib.add_log_level,
#         structlog.stdlib.PositionalArgumentsFormatter(),
#         structlog.processors.TimeStamper(fmt="iso"),
#         structlog.processors.StackInfoRenderer(),
#         structlog.processors.format_exc_info,
#         structlog.processors.UnicodeDecoder(),
#         structlog.processors.JSONRenderer(),
#     ],
#     context_class=dict,
#     logger_factory=structlog.stdlib.LoggerFactory(),
#     wrapper_class=structlog.stdlib.BoundLogger,
#     cache_logger_on_first_use=True,
# )

    def get_processor_chain(enable_dev_processors: bool = False) -> list[Processor]:
        """Build the processor chain based on environment needs.

        Args:
            enable_dev_processors: Whether to include development-specific processors

        Returns:
            List of configured processors
        """
        # Core processors - always included
        processors: list[Processor] = [
            # Context management
            # ------------------------------------
            # Use this as your first processor in :func:`structlog.configure` to ensure
            # context-local context is included in all log calls.
            structlog.contextvars.merge_contextvars,
            # ------------------------------------

            # Add log level
            # Add the log level to the event dict under the ``level`` key.

            # Since that's just the log method name, this processor works with non-stdlib
            # logging as well. Therefore it's importable both from `structlog.processors`
            # as well as from `structlog.stdlib`.
            structlog.processors.add_log_level, # pyright: ignore[reportAttributeAccessIssue]
            structlog.stdlib.add_logger_name,

            structlog.stdlib.PositionalArgumentsFormatter(),

            # Stack info renderer
            # Add stack information with key ``stack`` if ``stack_info`` is `True`.
            # Useful when you want to attach a stack dump to a log entry without
            # involving an exception and works analogously to the *stack_info* argument
            # of the Python standard library logging.
            structlog.processors.StackInfoRenderer(),
            # ------------------------------------

            # Set exc info
            # Set ``event_dict["exc_info"] = True`` if *method_name* is ``"exception"``.
            structlog.dev.set_exc_info,

            # Unicode decoder
            # Decode the event dict values from bytes to strings.
            structlog.processors.UnicodeDecoder(),
            # ------------------------------------

            # Time stamper
            # Add a timestamp to the event dict under the ``timestamp`` key.
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            # ------------------------------------

            # Console renderer
            # Render the event dict as a human-readable string.
            #  Render ``event_dict`` nicely aligned, possibly in colors, and ordered.

            # # If ``event_dict`` contains a true-ish ``exc_info`` key, it will be rendered
            # # *after* the log line. If Rich_ or better-exceptions_ are present, in colors
            # # and with extra context.
            # structlog.dev.ConsoleRenderer(),
            # ------------------------------------

            # # Basic event enrichment
            # structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
            # structlog.stdlib.add_log_level, # pyright: ignore[reportAttributeAccessIssue]
            # structlog.stdlib.add_logger_name,

            # # Custom context processors
            # _add_module_name,
            # _add_process_info,

            # # Error and stack trace handling
            # structlog.processors.StackInfoRenderer(),
            # structlog.processors.format_exc_info,

            # # Encoding
            # structlog.processors.UnicodeDecoder(),
        ]

        # # Development-specific processors
        # if enable_dev_processors:
        #     processors.append(
        #         structlog.processors.CallsiteParameterAdder(
        #             {
        #                 structlog.processors.CallsiteParameter.FILENAME,
        #                 structlog.processors.CallsiteParameter.FUNC_NAME,
        #                 structlog.processors.CallsiteParameter.LINENO,
        #                 structlog.processors.CallsiteParameter.MODULE,
        #                 structlog.processors.CallsiteParameter.THREAD,
        #                 structlog.processors.CallsiteParameter.THREAD_NAME,
        #                 structlog.processors.CallsiteParameter.PROCESS,
        #                 structlog.processors.CallsiteParameter.PROCESS_NAME,
        #             }
        #         )
        #     )

        return processors

    # Get shared processors
    shared_processors = get_processor_chain(not enable_json_logs)

    # Performance optimization: only add expensive processors when needed
    if not enable_json_logs:  # Development mode
        shared_processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.PATHNAME,
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.THREAD,
                    structlog.processors.CallsiteParameter.THREAD_NAME,
                    structlog.processors.CallsiteParameter.PROCESS,
                    structlog.processors.CallsiteParameter.PROCESS_NAME,
                }
            )
        )

    def get_renderer(enable_json_logs: bool) -> Processor:
        """Get the appropriate renderer based on environment.

        Args:
            enable_json_logs: Whether to use JSON logging

        Returns:
            Configured renderer processor
        """
        if enable_json_logs:
            return structlog.processors.JSONRenderer(
                sort_keys=False,
                serializer=lambda obj: str(obj)  # Simple serializer that handles non-JSON types
            )

        return structlog.dev.ConsoleRenderer(
            colors=True,
            # exception_formatter=structlog.dev.plain_traceback,
            sort_keys=True,
            # level_styles={
            #     'debug': structlog.dev.BLUE,
            #     'info': structlog.dev.GREEN,
            #     'warning': structlog.dev.YELLOW,
            #     'error': structlog.dev.RED,
            #     'critical': structlog.dev.RED,
            #     "notset": structlog.dev.RESET,
            # }
        )

    # Get the appropriate renderer
    renderer = get_renderer(enable_json_logs)
    if enable_json_logs:
        shared_processors.append(structlog.processors.format_exc_info)

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        # wrapper_class=structlog.stdlib.BoundLogger,  # Use regular BoundLogger by default
        wrapper_class=structlog.make_filtering_bound_logger(get_log_level(log_level)),
        # cache_logger_on_first_use=True,
    )

    return structlog.get_config()

    # # Configure standard logging
    # logging.config.dictConfig({
    #     "version": 1,
    #     "disable_existing_loggers": False,
    #     "formatters": {
    #         "structured": {
    #             "()": structlog.stdlib.ProcessorFormatter,
    #             "processor": renderer,
    #             "foreign_pre_chain": shared_processors,
    #         },
    #     },
    #     "handlers": {
    #         "default": {
    #             "level": log_level,
    #             "class": "logging.StreamHandler",
    #             "formatter": "structured",
    #             "stream": sys.stdout,  # Explicitly use stdout
    #         },
    #         "error": {
    #             "level": "ERROR",
    #             "class": "logging.StreamHandler",
    #             "formatter": "structured",
    #             "stream": sys.stderr,  # Errors go to stderr
    #         },
    #         "null": {
    #             "class": "logging.NullHandler",
    #         },
    #     },
    #     "loggers": {
    #         "": {
    #             "handlers": ["default", "error"],
    #             "level": log_level,
    #             "propagate": True,
    #         },
    #         **{
    #             name: {
    #                 "handlers": ["default", "error"],
    #                 "level": level,
    #                 "propagate": False,
    #             }
    #             for name, level in default_third_party.items()
    #         },
    #     },
    # })


def get_logger(name: str) -> BoundLogger:
    """Get a configured logger instance.

    Args:
        name: The name for the logger, typically __name__

    Returns:
        A configured structlog logger instance
    """
    return structlog.get_logger(name)


# Initialize default logger
# logger = get_logger(__name__)


if __name__ == "__main__":
    import rich
    # Example usage and testing
    logger = structlog.get_logger(__name__)
    configure_logging(enable_json_logs=False, log_level="DEBUG")
    logger = structlog.get_logger(__name__)
    rich.print(structlog.get_config())

    import bpdb
    bpdb.set_trace()

    def worker_process(name: str) -> None:
        """Example worker process function.

        Args:
            name: Name of the worker
        """
        proc_logger = get_logger(f"worker.{name}")

        # Clear context for this process
        clear_contextvars()

        # Bind process-specific context
        bind_contextvars(
            worker_name=name,
            worker_type="example"
        )

        proc_logger.info(f"Worker {name} starting")

        try:
            # Simulate some work
            1/0
        except Exception as e:
            proc_logger.error("Worker error", exc_info=e)

        proc_logger.info(f"Worker {name} finished")

    # Test in main process
    logger = get_logger("test_logger")
    clear_contextvars()

    # Bind some context variables that will be included in all log entries
    bind_contextvars(
        app_version="1.0.0",
        environment="development"
    )

    # Start some worker processes
    processes = []
    for i in range(3):
        p = multiprocessing.Process(
            target=worker_process,
            args=(f"worker_{i}",)
        )
        processes.append(p)
        p.start()

    # Test structured logging while workers run
    logger.info("Main process running",
                num_workers=len(processes))

    # Wait for all processes
    for p in processes:
        p.join()

    logger.info("All workers finished")


    # Clear context at the end
    clear_contextvars()
