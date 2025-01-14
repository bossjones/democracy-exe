"""
This type stub file was generated by pyright.
"""

import logging
from .typing import EventDict

"""
Extracted log level data used by both stdlib and native log level filters.
"""
CRITICAL = ...
FATAL = ...
ERROR = ...
WARNING = ...
WARN = ...
INFO = ...
DEBUG = ...
NOTSET = ...
NAME_TO_LEVEL = ...
LEVEL_TO_NAME = ...
_LEVEL_TO_NAME = ...
_NAME_TO_LEVEL = ...
def map_method_name(method_name: str) -> str:
    ...

def add_log_level(logger: logging.Logger, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add the log level to the event dict under the ``level`` key.

    Since that's just the log method name, this processor works with non-stdlib
    logging as well. Therefore it's importable both from `structlog.processors`
    as well as from `structlog.stdlib`.

    .. versionadded:: 15.0.0
    .. versionchanged:: 20.2.0
       Importable from `structlog.processors` (additionally to
       `structlog.stdlib`).
    .. versionchanged:: 24.1.0
       Added mapping from "exception" to "error"
    """
    ...

