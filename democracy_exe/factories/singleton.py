"""Singleton metaclass for creating singleton classes."""
from __future__ import annotations

from typing import Any


class Singleton(type):
    """Metaclass for creating singleton classes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the singleton.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.__instance = None
        super().__init__(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Create or return the singleton instance.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            Any: The singleton instance
        """
        if self.__instance is None:
            self.__instance = super().__call__(*args, **kwargs)
        return self.__instance
