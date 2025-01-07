from __future__ import annotations

import logging
import logging.config

import structlog

from democracy_exe.aio_settings import aiosettings


# NOTE: if you want to try out setup_logging, comment this out
custom_logger = structlog.stdlib.get_logger("custom_logger")

# TODO: Try out both of these


# SOURCE: https://github.com/thijsmie/rollbot/blob/master/src/rollbot/logsetup.py
def setup_logging() -> None:
    """Configure structured logging for the application using structlog and standard logging.

    This function sets up a comprehensive logging system that:
    1. Configures both structlog and standard logging
    2. Sets up JSON and colored console formatters
    3. Configures specific loggers for discord-related modules
    4. Adds various processors for enriching log messages
    """
    # Create a timestamper processor that adds ISO-format UTC timestamps to log entries
    timestamper = structlog.processors.TimeStamper(fmt="ISO", utc=True)

    # Define pre-chain processors that run before log messages hit the formatter
    # These processors add context like log level, timestamps, and code location
    pre_chain = [
        structlog.stdlib.add_log_level,  # Adds log level to the event dict
        structlog.stdlib.ExtraAdder(),   # Allows passing extra values to log methods
        timestamper,                     # Adds timestamp to the event dict
        # Adds call site information (file, function, line number) to log entries
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.FILENAME,
                structlog.processors.CallsiteParameter.FUNC_NAME,
                structlog.processors.CallsiteParameter.LINENO,
                structlog.processors.CallsiteParameter.MODULE,
            }
        ),
    ]

    # Configure the Python standard logging system using a dictionary configuration
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,  # Preserve existing loggers
            "formatters": {
                # JSON formatter for structured logging output
                "default": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.processors.JSONRenderer(),  # Outputs logs as JSON
                    ],
                    "foreign_pre_chain": pre_chain,
                },
                # Colored console formatter for development
                "colored": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processors": [
                        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                        structlog.dev.ConsoleRenderer(colors=True),  # Colored output
                    ],
                    "foreign_pre_chain": pre_chain,
                },
            },
            # Define handlers that determine where logs are sent
            "handlers": {
                "default": {
                    "level": "INFO",
                    "class": "logging.StreamHandler",  # Outputs to stderr
                    "formatter": "default",
                },
            },
            # Configure specific loggers and their properties
            "loggers": {
                # Root logger configuration
                "": {
                    "handlers": ["default"],
                    "level": "INFO",
                    "propagate": True,
                },
                # Discord-specific logger configurations to reduce noise
                "discord": {
                    "level": "WARNING",
                    "propagate": True,
                },
                "discord.client": {
                    "level": "WARNING",
                    "propagate": True,
                },
                "discord.gateway": {
                    "level": "WARNING",
                    "propagate": True,
                },
            },
        }
    )

    # Configure structlog with processors that determine how structured logging works
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,                    # Add log level
            structlog.stdlib.PositionalArgumentsFormatter(),   # Handle args
            timestamper,                                       # Add timestamps
            structlog.processors.StackInfoRenderer(),          # Add stack info
            structlog.processors.format_exc_info,              # Format exceptions
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,  # Wrap for formatter
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),      # Use stdlib's logger
        wrapper_class=structlog.stdlib.BoundLogger,           # Use bound logger
        # cache_logger_on_first_use=True,                       # Cache for performance
    )

# SOURCE: https://www.perplexity.ai/search/similar-to-loguru-s-concept-of-W0mcX6HRRRiRgi7qT.K1bQ
# Intercept logging.Handler to use structlog
class StructLogHandler(logging.Handler):
    def emit(self, record):
        # Get the structlog logger
        logger = structlog.get_logger(record.name)

        # Map the record to structlog's log level
        if record.levelno <= logging.DEBUG:
            log_method = logger.debug
        elif record.levelno <= logging.INFO:
            log_method = logger.info
        elif record.levelno <= logging.WARNING:
            log_method = logger.warning
        elif record.levelno <= logging.ERROR:
            log_method = logger.error
        else:
            log_method = logger.critical

        # Log the message
        log_method(
            record.getMessage(),
            exc_info=record.exc_info,
            stack_info=record.stack_info,
        )

# set up logging to use structlog
logging.basicConfig(handlers=[StructLogHandler()], level=aiosettings.log_level)



# SOURCE: https://gist.github.com/nkhitrov/38adbb314f0d35371eba4ffb8f27078f
def configure_logger(enable_json_logs: bool = False) -> None:
    """Configure structured logging with optional JSON output format.

    This function sets up a comprehensive logging system optimized for async applications
    with detailed context information and flexible output formatting.

    Args:
        enable_json_logs: If True, outputs logs in JSON format. Otherwise, uses colored console output.
    """
    # Create a timestamper that adds human-readable timestamps to log entries
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")

    # Define processors that will be shared between structlog and standard logging
    # These processors enrich log entries with contextual information
    shared_processors = [
        timestamper,                                     # Add timestamps
        structlog.stdlib.add_log_level,                  # Add log level
        structlog.stdlib.add_logger_name,                # Add logger name
        structlog.contextvars.merge_contextvars,         # Add context variables
        # Add extensive call site information including thread and process details
        structlog.processors.CallsiteParameterAdder(
            {
                structlog.processors.CallsiteParameter.PATHNAME,      # Full path
                structlog.processors.CallsiteParameter.FILENAME,      # File name
                structlog.processors.CallsiteParameter.MODULE,        # Module name
                structlog.processors.CallsiteParameter.FUNC_NAME,     # Function name
                structlog.processors.CallsiteParameter.THREAD,        # Thread ID
                structlog.processors.CallsiteParameter.THREAD_NAME,   # Thread name
                structlog.processors.CallsiteParameter.PROCESS,       # Process ID
                structlog.processors.CallsiteParameter.PROCESS_NAME,  # Process name
            }
        ),
        structlog.stdlib.ExtraAdder(),                   # Allow adding extra fields
    ]

    # Configure structlog with async support
    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),      # Use stdlib's logger
        # call log with await syntax in thread pool executor
        wrapper_class=structlog.stdlib.AsyncBoundLogger,      # Enable async logging
        # cache_logger_on_first_use=True,                      # Cache for performance
    )

    # Choose the renderer based on enable_json_logs flag
    logs_render = (
        structlog.processors.JSONRenderer()                   # JSON output
        if enable_json_logs
        else structlog.dev.ConsoleRenderer(colors=True)      # Colored console output
    )

    # Configure the default logging with custom formatters
    _configure_default_logging_by_custom(shared_processors, logs_render)


def _configure_default_logging_by_custom(
    shared_processors: list,
    logs_render: structlog.typing.Processor
) -> None:
    """Configure the default logging system with custom processors and renderer.

    Args:
        shared_processors: List of processors to be applied to all log entries
        logs_render: The renderer to use for formatting log output
    """
    # Create a handler that writes to stderr
    handler = logging.StreamHandler()

    # Use `ProcessorFormatter` to format all `logging` entries.
    # Create a formatter that processes both structlog and standard logging entries
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,                  # Processors for non-structlog logs
        processors=[
            _extract_from_record,                            # Extract thread/process info
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,  # Clean metadata
            logs_render,                                     # Render the log entry
        ],
    )

    # Set up the root logger with the custom handler and formatter
    handler.setFormatter(formatter)
    root_uvicorn_logger = logging.getLogger()
    root_uvicorn_logger.addHandler(handler)
    root_uvicorn_logger.setLevel(aiosettings.log_level)


def _extract_from_record(
    _: str,
    __: str,
    event_dict: dict
) -> dict:
    """Extract thread and process names from the log record and add to event dict.

    Args:
        _: Unused logger name
        __: Unused log level name
        event_dict: The dictionary containing the log event data

    Returns:
        dict: The event dictionary enriched with thread and process names
    """
    # Extract the log record from the event dictionary
    record = event_dict["_record"]
    # Add thread and process names to the event dictionary
    event_dict["thread_name"] = record.threadName
    event_dict["process_name"] = record.processName
    return event_dict

if __name__ == "__main__":
    configure_logger()

    logger = structlog.stdlib.get_logger(__name__)
    logger.info("Hello, world!")
