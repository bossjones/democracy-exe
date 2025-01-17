"""Base service class for all services."""
from __future__ import annotations

import logging

from typing import Any, Optional

import structlog


logger = structlog.get_logger(__name__)


class BaseService:
    """Base class for all services."""

    def __init__(self, bot: Any = None, service_logger: logging.Logger | None = None) -> None:
        """Initialize the service.

        Args:
            bot: The Discord bot instance
            logger: Optional logger instance
        """
        self.bot = bot
        self._logger = service_logger or logger.bind(service=self.__class__.__name__)

    @property
    def logger(self) -> logging.Logger:
        """Get the logger instance."""
        return self._logger
