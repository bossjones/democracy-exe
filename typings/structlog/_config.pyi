"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, Iterable, Sequence
from .typing import BindableLogger, Context, Processor, WrappedLogger

"""
Global state department.  Don't reload this module or everything breaks.
"""
_BUILTIN_DEFAULT_PROCESSORS: Sequence[Processor] = ...
_BUILTIN_DEFAULT_CONTEXT_CLASS = ...
_BUILTIN_DEFAULT_WRAPPER_CLASS = ...
_BUILTIN_DEFAULT_LOGGER_FACTORY = ...
_BUILTIN_CACHE_LOGGER_ON_FIRST_USE = ...
class _Configuration:
    """
    Global defaults.
    """
    is_configured: bool = ...
    default_processors: Iterable[Processor] = ...
    default_context_class: type[Context] = ...
    default_wrapper_class: Any = ...
    logger_factory: Callable[..., WrappedLogger] = ...
    cache_logger_on_first_use: bool = ...


_CONFIG = ...
def is_configured() -> bool:
    """
    Return whether *structlog* has been configured.

    If `False`, *structlog* is running with builtin defaults.

    .. versionadded: 18.1.0
    """
    ...

def get_config() -> dict[str, Any]:
    """
    Get a dictionary with the current configuration.

    .. note::

       Changes to the returned dictionary do *not* affect *structlog*.

    .. versionadded: 18.1.0
    """
    ...

def get_logger(*args: Any, **initial_values: Any) -> Any:
    """
    Convenience function that returns a logger according to configuration.

    >>> from structlog import get_logger
    >>> log = get_logger(y=23)
    >>> log.info("hello", x=42)
    y=23 x=42 event='hello'

    Args:
        args:
            *Optional* positional arguments that are passed unmodified to the
            logger factory.  Therefore it depends on the factory what they
            mean.

        initial_values: Values that are used to pre-populate your contexts.

    Returns:
        A proxy that creates a correctly configured bound logger when
        necessary. The type of that bound logger depends on your configuration
        and is `structlog.BoundLogger` by default.

    See `configuration` for details.

    If you prefer CamelCase, there's an alias for your reading pleasure:
    `structlog.getLogger`.

    .. versionadded:: 0.4.0 *args*
    """
    ...

getLogger = ...
def wrap_logger(logger: WrappedLogger | None, processors: Iterable[Processor] | None = ..., wrapper_class: type[BindableLogger] | None = ..., context_class: type[Context] | None = ..., cache_logger_on_first_use: bool | None = ..., logger_factory_args: Iterable[Any] | None = ..., **initial_values: Any) -> Any:
    """
    Create a new bound logger for an arbitrary *logger*.

    Default values for *processors*, *wrapper_class*, and *context_class* can
    be set using `configure`.

    If you set an attribute here, `configure` calls have *no* effect for the
    *respective* attribute.

    In other words: selective overwriting of the defaults while keeping some
    *is* possible.

    Args:
        initial_values: Values that are used to pre-populate your contexts.

        logger_factory_args:
            Values that are passed unmodified as ``*logger_factory_args`` to
            the logger factory if not `None`.

    Returns:
        A proxy that creates a correctly configured bound logger when
        necessary.

    See `configure` for the meaning of the rest of the arguments.

    .. versionadded:: 0.4.0 *logger_factory_args*
    """
    ...

def configure(processors: Iterable[Processor] | None = ..., wrapper_class: type[BindableLogger] | None = ..., context_class: type[Context] | None = ..., logger_factory: Callable[..., WrappedLogger] | None = ..., cache_logger_on_first_use: bool | None = ...) -> None:
    """
    Configures the **global** defaults.

    They are used if `wrap_logger` or `get_logger` are called without
    arguments.

    Can be called several times, keeping an argument at `None` leaves it
    unchanged from the current setting.

    After calling for the first time, `is_configured` starts returning `True`.

    Use `reset_defaults` to undo your changes.

    Args:
        processors: The processor chain. See :doc:`processors` for details.

        wrapper_class:
            Class to use for wrapping loggers instead of
            `structlog.BoundLogger`.  See `standard-library`, :doc:`twisted`,
            and `custom-wrappers`.

        context_class:
            Class to be used for internal context keeping. The default is a
            `dict` and since dictionaries are ordered as of Python 3.6, there's
            few reasons to change this option.

        logger_factory:
            Factory to be called to create a new logger that shall be wrapped.

        cache_logger_on_first_use:
            `wrap_logger` doesn't return an actual wrapped logger but a proxy
            that assembles one when it's first used. If this option is set to
            `True`, this assembled logger is cached. See `performance`.

    .. versionadded:: 0.3.0 *cache_logger_on_first_use*
    """
    ...

def configure_once(processors: Iterable[Processor] | None = ..., wrapper_class: type[BindableLogger] | None = ..., context_class: type[Context] | None = ..., logger_factory: Callable[..., WrappedLogger] | None = ..., cache_logger_on_first_use: bool | None = ...) -> None:
    """
    Configures if structlog isn't configured yet.

    It does *not* matter whether it was configured using `configure` or
    `configure_once` before.

    Raises:
        RuntimeWarning: if repeated configuration is attempted.
    """
    ...

def reset_defaults() -> None:
    """
    Resets global default values to builtin defaults.

    `is_configured` starts returning `False` afterwards.
    """
    ...

class BoundLoggerLazyProxy:
    """
    Instantiates a bound logger on first usage.

    Takes both configuration and instantiation parameters into account.

    The only points where a bound logger changes state are ``bind()``,
    ``unbind()``, and ``new()`` and that return the actual ``BoundLogger``.

    If and only if configuration says so, that actual bound logger is cached on
    first usage.

    .. versionchanged:: 0.4.0 Added support for *logger_factory_args*.
    """
    def __init__(self, logger: WrappedLogger | None, wrapper_class: type[BindableLogger] | None = ..., processors: Iterable[Processor] | None = ..., context_class: type[Context] | None = ..., cache_logger_on_first_use: bool | None = ..., initial_values: dict[str, Any] | None = ..., logger_factory_args: Any = ...) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def bind(self, **new_values: Any) -> BindableLogger:
        """
        Assemble a new BoundLogger from arguments and configuration.
        """
        ...
    
    def unbind(self, *keys: str) -> BindableLogger:
        """
        Same as bind, except unbind *keys* first.

        In our case that could be only initial values.
        """
        ...
    
    def try_unbind(self, *keys: str) -> BindableLogger:
        ...
    
    def new(self, **new_values: Any) -> BindableLogger:
        """
        Clear context, then bind.
        """
        ...
    
    def __getattr__(self, name: str) -> Any:
        """
        If a logging method if called on a lazy proxy, we have to create an
        ephemeral BoundLogger first.
        """
        ...
    
    def __getstate__(self) -> dict[str, Any]:
        """
        Our __getattr__ magic makes this necessary.
        """
        ...
    
    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Our __getattr__ magic makes this necessary.
        """
        ...
    

