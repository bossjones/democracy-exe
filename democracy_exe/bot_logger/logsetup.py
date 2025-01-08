from __future__ import annotations

import logging
import logging.config
import multiprocessing
import sys

from typing import Any, Dict, List, Optional

import structlog

from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars, reset_contextvars
from structlog.stdlib import BoundLogger
from structlog.types import Processor

from democracy_exe.aio_settings import aiosettings


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
) -> None:
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

    # Performance optimization: pre-configure shared processors
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True)

    # Create processors that will be shared between structlog and standard logging
    shared_processors: list[Processor] = [
        # Merge context vars should be first to ensure context is available to other processors
        merge_contextvars,
        timestamper,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_module_name,  # Add custom module name processor
        _add_process_info,  # Add process info for multiprocessing support
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Performance optimization: only add expensive processors when needed
    if not enable_json_logs:  # Development mode
        shared_processors.append(
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.MODULE,
                    structlog.processors.CallsiteParameter.THREAD,
                    structlog.processors.CallsiteParameter.THREAD_NAME,
                    structlog.processors.CallsiteParameter.PROCESS,
                    structlog.processors.CallsiteParameter.PROCESS_NAME,
                }
            )
        )

    # Performance optimization for production
    if enable_json_logs:
        # JSON output optimized for production
        renderer = structlog.processors.JSONRenderer(serializer=structlog.processors.JSONRenderer.get_default_serializer(sort_keys=False))
        shared_processors.append(structlog.processors.format_exc_info)
    else:
        # Pretty output for development
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
            sort_keys=True
        )

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.AsyncBoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": renderer,
                "foreign_pre_chain": shared_processors,
            },
        },
        "handlers": {
            "default": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "structured",
                "stream": sys.stdout,  # Explicitly use stdout
            },
            "error": {
                "level": "ERROR",
                "class": "logging.StreamHandler",
                "formatter": "structured",
                "stream": sys.stderr,  # Errors go to stderr
            },
            "null": {
                "class": "logging.NullHandler",
            },
        },
        "loggers": {
            "": {
                "handlers": ["default", "error"],
                "level": log_level,
                "propagate": True,
            },
            **{
                name: {
                    "handlers": ["default", "error"],
                    "level": level,
                    "propagate": False,
                }
                for name, level in default_third_party.items()
            },
        },
    })


def get_logger(name: str) -> BoundLogger:
    """Get a configured logger instance.

    Args:
        name: The name for the logger, typically __name__

    Returns:
        A configured structlog logger instance
    """
    return structlog.get_logger(name)


# Initialize default logger
logger = get_logger(__name__)


if __name__ == "__main__":
    # Example usage and testing
    configure_logging(enable_json_logs=False, log_level="DEBUG")

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
