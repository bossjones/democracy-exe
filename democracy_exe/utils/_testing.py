# SOURCE: https://github.com/roocs/clisops/blob/dd87a2e335b8f06d54e2a15ae67ab3e79609ec77/clisops/utils/testing.py#L417
from __future__ import annotations

from typing import Any, Optional, Type


class ContextLogger:
    """Helper class for safe logging management in pytests.

    This context manager handles enabling and disabling loggers during pytest execution,
    with special handling for pytest's caplog fixture.

    Args:
        caplog: Whether pytest's caplog fixture is being used.
    """
    import loguru


    logger: loguru.Logger
    using_caplog: bool
    _package: str

    def __init__(self, caplog: bool = False) -> None:
        """Initialize the ContextLogger.

        Args:
            caplog: Flag indicating if pytest's caplog fixture is being used.
        """
        import structlog

logger = structlog.get_logger(__name__)

        self.logger = logger
        self.using_caplog = False
        self._package = ""
        if caplog:
            self.using_caplog = True

    def __enter__(self, package_name: str = "clisops") -> loguru.Logger:
        """Enter the context manager.

        Args:
            package_name: Name of the package to enable logging for.

        Returns:
            The configured logger instance.
        """
        self.logger.enable(package_name)
        self._package = package_name
        return self.logger

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> None:
        """Exit the context manager.

        If test is supplying caplog, pytest will manage teardown.

        Args:
            exc_type: The type of the exception that was raised, if any.
            exc_val: The instance of the exception that was raised, if any.
            exc_tb: The traceback of the exception that was raised, if any.
        """
        self.logger.disable(self._package)
        if not self.using_caplog:
            try: # noqa: SIM105
                self.logger.remove()
            except ValueError:
                pass
