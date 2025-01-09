"""
This type stub file was generated by pyright.
"""

from types import TracebackType
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Protocol, TextIO, Tuple, Type, Union, runtime_checkable

"""
Type information used throughout *structlog*.

For now, they are considered provisional. Especially `BindableLogger` will
probably change to something more elegant.

.. versionadded:: 22.2.0
"""
WrappedLogger = Any
Context = Union[Dict[str, Any], Dict[Any, Any]]
EventDict = MutableMapping[str, Any]
ProcessorReturnValue = Union[Mapping[str, Any], str, bytes, bytearray, Tuple[Any, ...]]
Processor = Callable[[WrappedLogger, str, EventDict], ProcessorReturnValue]
ExcInfo = Tuple[Type[BaseException], BaseException, Optional[TracebackType]]
ExceptionRenderer = Callable[[TextIO, ExcInfo], None]
@runtime_checkable
class ExceptionTransformer(Protocol):
    """
    **Protocol:** A callable that transforms an `ExcInfo` into another
    datastructure.

    The result should be something that your renderer can work with, e.g., a
    ``str`` or a JSON-serializable ``dict``.

    Used by `structlog.processors.format_exc_info()` and
    `structlog.processors.ExceptionPrettyPrinter`.

    Args:
        exc_info: Is the exception tuple to format

    Returns:
        Anything that can be rendered by the last processor in your chain, for
        example, a string or a JSON-serializable structure.

    .. versionadded:: 22.1.0
    """
    def __call__(self, exc_info: ExcInfo) -> Any:
        ...
    


@runtime_checkable
class BindableLogger(Protocol):
    """
    **Protocol**: Methods shared among all bound loggers and that are relied on
    by *structlog*.

    .. versionadded:: 20.2.0
    """
    _context: Context
    def bind(self, **new_values: Any) -> BindableLogger:
        ...
    
    def unbind(self, *keys: str) -> BindableLogger:
        ...
    
    def try_unbind(self, *keys: str) -> BindableLogger:
        ...
    
    def new(self, **new_values: Any) -> BindableLogger:
        ...
    


class FilteringBoundLogger(BindableLogger, Protocol):
    """
    **Protocol**: A `BindableLogger` that filters by a level.

    The only way to instantiate one is using `make_filtering_bound_logger`.

    .. versionadded:: 20.2.0
    .. versionadded:: 22.2.0 String interpolation using positional arguments.
    .. versionadded:: 22.2.0
       Async variants ``alog()``, ``adebug()``, ``ainfo()``, and so forth.
    .. versionchanged:: 22.3.0
       String interpolation is only attempted if positional arguments are
       passed.
    """
    def bind(self, **new_values: Any) -> FilteringBoundLogger:
        """
        Return a new logger with *new_values* added to the existing ones.

        .. versionadded:: 22.1.0
        """
        ...
    
    def unbind(self, *keys: str) -> FilteringBoundLogger:
        """
        Return a new logger with *keys* removed from the context.

        .. versionadded:: 22.1.0
        """
        ...
    
    def try_unbind(self, *keys: str) -> FilteringBoundLogger:
        """
        Like :meth:`unbind`, but best effort: missing keys are ignored.

        .. versionadded:: 22.1.0
        """
        ...
    
    def new(self, **new_values: Any) -> FilteringBoundLogger:
        """
        Clear context and binds *initial_values* using `bind`.

        .. versionadded:: 22.1.0
        """
        ...
    
    def debug(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **debug** level.
        """
        ...
    
    async def adebug(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **debug** level.

        ..versionadded:: 22.2.0
        """
        ...
    
    def info(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **info** level.
        """
        ...
    
    async def ainfo(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **info** level.

        ..versionadded:: 22.2.0
        """
        ...
    
    def warning(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **warn** level.
        """
        ...
    
    async def awarning(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **warn** level.

        ..versionadded:: 22.2.0
        """
        ...
    
    def warn(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **warn** level.
        """
        ...
    
    async def awarn(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **warn** level.

        ..versionadded:: 22.2.0
        """
        ...
    
    def error(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **error** level.
        """
        ...
    
    async def aerror(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **error** level.

        ..versionadded:: 22.2.0
        """
        ...
    
    def err(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **error** level.
        """
        ...
    
    def fatal(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **critical** level.
        """
        ...
    
    async def afatal(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **critical** level.

        ..versionadded:: 22.2.0
        """
        ...
    
    def exception(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **error** level and ensure that
        ``exc_info`` is set in the event dictionary.
        """
        ...
    
    async def aexception(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **error** level and ensure that
        ``exc_info`` is set in the event dictionary.

        ..versionadded:: 22.2.0
        """
        ...
    
    def critical(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **critical** level.
        """
        ...
    
    async def acritical(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **critical** level.

        ..versionadded:: 22.2.0
        """
        ...
    
    def msg(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **info** level.
        """
        ...
    
    async def amsg(self, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at **info** level.
        """
        ...
    
    def log(self, level: int, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at *level*.
        """
        ...
    
    async def alog(self, level: int, event: str, *args: Any, **kw: Any) -> Any:
        """
        Log ``event % args`` with **kw** at *level*.
        """
        ...
    


