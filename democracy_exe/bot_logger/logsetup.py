from __future__ import annotations

import logging
import logging.config
import sys

from typing import Any, Dict, List, Optional

import structlog

from structlog.contextvars import bind_contextvars, clear_contextvars, merge_contextvars, reset_contextvars
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

    # Create processors that will be shared between structlog and standard logging
    shared_processors: list[Processor] = [
        # Merge context vars should be first to ensure context is available to other processors
        merge_contextvars,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=True),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _add_module_name,  # Add custom module name processor
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
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
        ),
    ]

    # Performance optimization for production
    if enable_json_logs:
        # JSON output optimized for production
        renderer = structlog.processors.JSONRenderer()
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


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
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
    logger = get_logger("test_logger")

    # Clear any existing context at the start of a new request/context
    clear_contextvars()

    # Bind some context variables that will be included in all log entries
    bind_contextvars(
        app_version="1.0.0",
        environment="development"
    )

    # Test basic logging with context
    logger.info("Test message", extra={"key": "value"})

    # Test context override
    tokens = bind_contextvars(request_id="123")
    try:
        logger.info("Processing request")
        # Test error logging with exception
        try:
            1/0
        except Exception as e:
            logger.error("Error occurred", exc_info=e)
    finally:
        # Restore previous context
        reset_contextvars(**tokens)

    # Test structured logging with global context
    logger.info("Structured log",
                user_id=123,
                action="login",
                status="success",
                duration_ms=532)

    # Clear context at the end
    clear_contextvars()
