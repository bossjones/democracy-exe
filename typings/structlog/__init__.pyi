"""
This type stub file was generated by pyright.
"""

from structlog import contextvars, dev, processors, stdlib, testing, threadlocal, tracebacks, twisted, types, typing
from structlog._base import BoundLoggerBase, get_context
from structlog._config import configure, configure_once, getLogger, get_config, get_logger, is_configured, reset_defaults, wrap_logger
from structlog._generic import BoundLogger
from structlog._native import make_filtering_bound_logger
from structlog._output import BytesLogger, BytesLoggerFactory, PrintLogger, PrintLoggerFactory, WriteLogger, WriteLoggerFactory
from structlog.exceptions import DropEvent
from structlog.testing import ReturnLogger, ReturnLoggerFactory

__title__ = ...
__author__ = ...
__license__ = ...
__copyright__ = ...
__all__ = ["BoundLogger", "BoundLoggerBase", "BytesLogger", "BytesLoggerFactory", "configure_once", "configure", "contextvars", "dev", "DropEvent", "get_config", "get_context", "get_logger", "getLogger", "is_configured", "make_filtering_bound_logger", "PrintLogger", "PrintLoggerFactory", "processors", "reset_defaults", "ReturnLogger", "ReturnLoggerFactory", "stdlib", "testing", "threadlocal", "tracebacks", "twisted", "types", "typing", "wrap_logger", "WriteLogger", "WriteLoggerFactory"]
def __getattr__(name: str) -> str:
    ...
