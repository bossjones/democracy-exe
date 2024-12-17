"""
This type stub file was generated by pyright.
"""

import sys
import logging
from . import util

COLORS = ...
COLORS_DEFAULT = ...
if util.WINDOWS:
    ANSI = ...
    OFFSET = ...
    CHAR_SKIP = ...
    CHAR_SUCCESS = ...
    CHAR_ELLIPSIES = ...
else:
    ANSI = ...
    OFFSET = ...
    CHAR_SKIP = ...
    CHAR_SUCCESS = ...
    CHAR_ELLIPSIES = ...
LOG_FORMAT = ...
LOG_FORMAT_DATE = ...
LOG_LEVEL = ...
LOG_LEVELS = ...
class Logger(logging.Logger):
    """Custom Logger that includes extra info in log records"""
    def makeRecord(self, name, level, fn, lno, msg, args, exc_info, func=..., extra=..., sinfo=..., factory=...):
        ...
    


class LoggerAdapter:
    """Trimmed-down version of logging.LoggingAdapter"""
    __slots__ = ...
    def __init__(self, logger, job) -> None:
        ...
    
    def debug(self, msg, *args, **kwargs): # -> None:
        ...
    
    def info(self, msg, *args, **kwargs): # -> None:
        ...
    
    def warning(self, msg, *args, **kwargs): # -> None:
        ...
    
    def error(self, msg, *args, **kwargs): # -> None:
        ...
    


class PathfmtProxy:
    __slots__ = ...
    def __init__(self, job) -> None:
        ...
    
    def __getattribute__(self, name): # -> Any | None:
        ...
    
    def __str__(self) -> str:
        ...
    


class KwdictProxy:
    __slots__ = ...
    def __init__(self, job) -> None:
        ...
    
    def __getattribute__(self, name): # -> Any | None:
        ...
    


class Formatter(logging.Formatter):
    """Custom formatter that supports different formats per loglevel"""
    def __init__(self, fmt, datefmt) -> None:
        ...
    
    def format(self, record): # -> Any | str | LiteralString | Literal[False]:
        ...
    


def initialize_logging(loglevel): # -> Logger:
    """Setup basic logging functionality before configfiles have been loaded"""
    ...

def configure_logging(loglevel): # -> None:
    ...

def setup_logging_handler(key, fmt=..., lvl=..., mode=...): # -> FileHandler | None:
    """Setup a new logging handler"""
    ...

def stdout_write_flush(s): # -> None:
    ...

def stderr_write_flush(s): # -> None:
    ...

if getattr(sys.stdout, "line_buffering", None):
    def stdout_write(s): # -> None:
        ...
    
else:
    stdout_write = ...
if getattr(sys.stderr, "line_buffering", None):
    def stderr_write(s): # -> None:
        ...
    
else:
    stderr_write = ...
def configure_standard_streams(): # -> None:
    ...

def select(): # -> ColorOutput | TerminalOutput | PipeOutput | CustomOutput:
    """Select a suitable output class"""
    ...

class NullOutput:
    def start(self, path): # -> None:
        """Print a message indicating the start of a download"""
        ...
    
    def skip(self, path): # -> None:
        """Print a message indicating that a download has been skipped"""
        ...
    
    def success(self, path): # -> None:
        """Print a message indicating the completion of a download"""
        ...
    
    def progress(self, bytes_total, bytes_downloaded, bytes_per_second): # -> None:
        """Display download progress"""
        ...
    


class PipeOutput(NullOutput):
    def skip(self, path): # -> None:
        ...
    
    def success(self, path): # -> None:
        ...
    


class TerminalOutput:
    def __init__(self) -> None:
        ...
    
    def start(self, path): # -> None:
        ...
    
    def skip(self, path): # -> None:
        ...
    
    def success(self, path): # -> None:
        ...
    
    def progress(self, bytes_total, bytes_downloaded, bytes_per_second): # -> None:
        ...
    


class ColorOutput(TerminalOutput):
    def __init__(self) -> None:
        ...
    
    def start(self, path): # -> None:
        ...
    
    def skip(self, path): # -> None:
        ...
    
    def success(self, path): # -> None:
        ...
    


class CustomOutput:
    def __init__(self, options) -> None:
        ...
    
    def start(self, path): # -> None:
        ...
    
    def skip(self, path): # -> None:
        ...
    
    def success(self, path): # -> None:
        ...
    
    def progress(self, bytes_total, bytes_downloaded, bytes_per_second): # -> None:
        ...
    


class EAWCache(dict):
    def __missing__(self, key): # -> Literal[2, 1]:
        ...
    


def shorten_string(txt, limit, sep=...):
    """Limit width of 'txt'; assume all characters have a width of 1"""
    ...

def shorten_string_eaw(txt, limit, sep=..., cache=...):
    """Limit width of 'txt'; check for east-asian characters with width > 1"""
    ...
