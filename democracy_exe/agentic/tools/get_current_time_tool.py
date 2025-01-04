# pyright: reportAttributeAccessIssue=false
"""Time retrieval tool for LangChain/LangGraph integration."""
from __future__ import annotations

import re

from datetime import datetime
from typing import Any, Dict, Optional, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field


class GetCurrentTimeInput(BaseModel):
    """Schema for current time tool input.

    Attributes:
        format: Optional datetime format string (defaults to "%Y-%m-%d %H:%M:%S")
    """
    format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Format string for datetime output"
    )


class GetCurrentTimeResponse(BaseModel):
    """Schema for current time tool response.

    Attributes:
        current_time: Formatted current time string
        timestamp: Unix timestamp
        error: Optional error message if operation failed
    """
    current_time: str = Field(description="Formatted current time string")
    timestamp: float = Field(description="Unix timestamp")
    error: str | None = Field(
        default=None,
        description="Error message if operation failed"
    )


class GetCurrentTimeTool(BaseTool):
    """Tool for retrieving the current time in various formats.

    This tool provides the current time with optional format specification.
    It supports both synchronous and asynchronous operations.

    Attributes:
        name: Tool name for LangChain
        description: Tool description for LLM context
        args_schema: Pydantic model for argument validation
        return_direct: Whether to return results directly to user
    """
    name: str = "current_time"
    description: str = """Get the current time in a specified format.
    Returns both formatted time string and Unix timestamp."""
    args_schema: type[BaseModel] = GetCurrentTimeInput
    return_direct: bool = True

    # def _get_time(self, format: str = "%Y-%m-%d %H:%M:%S") -> tuple[str, float]:
    #     """Get the current time in specified format.

    #     Args:
    #         format: Format string for datetime output

    #     Returns:
    #         tuple[str, float]: Tuple of (formatted time string, Unix timestamp)

    #     Raises:
    #         ValueError: If format string is invalid
    #     """
    #     try:
    #         # Validate format string by attempting to use it
    #         datetime.now().strftime(format)
    #     except Exception as e:
    #         logger.error(f"Error formatting time: {e!s}")
    #         raise ValueError(f"Invalid time format: {e!s}")

    #     now = datetime.now()
    #     logger.debug(f"Getting current time with format: {format}")
    #     formatted_time = now.strftime(format)
    #     timestamp = now.timestamp()
    #     logger.debug(f"Current time: {formatted_time}, Timestamp: {timestamp}")
    #     return formatted_time, timestamp


    def _get_time(self, format: str = "%Y-%m-%d %H:%M:%S") -> tuple[str, float]:
        """Get the current time in specified format.

        Args:
            format: Format string for datetime output

        Returns:
            tuple[str, float]: Tuple of (formatted time string, Unix timestamp)

        Raises:
            ValueError: If format string is invalid
        """
        logger.debug(f"Getting current time with format: {format}")

        try:
            now = datetime.now()
            # This will raise ValueError if format string is invalid
            formatted_time = now.strftime(format)
            timestamp = now.timestamp()

            logger.debug(f"Current time: {formatted_time}, Timestamp: {timestamp}")
            return formatted_time, timestamp

        except (ValueError, TypeError) as e:
            logger.error(f"Error formatting time: {e}")
            raise ValueError(f"Invalid time format: {e}")

    def _run(
        self,
        format: str = "%Y-%m-%d %H:%M:%S",
        run_manager: CallbackManagerForToolRun | None = None
    ) -> dict[str, Any]:
        """Get current time synchronously.

        Args:
            format: Format string for datetime output
            run_manager: Callback manager for tool execution

        Returns:
            Dict[str, Any]: Response containing current time information

        Raises:
            ValueError: If time formatting fails
        """
        logger.info("Getting current time synchronously")
        try:
            formatted_time, timestamp = self._get_time(format)
            response = GetCurrentTimeResponse(
                current_time=formatted_time,
                timestamp=timestamp
            ).model_dump()
            logger.info(f"Successfully retrieved current time: {formatted_time}")
            return response
        except Exception as e:
            logger.exception("Failed to get current time")
            error_response = GetCurrentTimeResponse(
                current_time="",
                timestamp=0.0,
                error=str(e)
            ).model_dump()
            logger.error(f"Returning error response: {error_response}")
            return error_response

    async def _arun(
        self,
        format: str = "%Y-%m-%d %H:%M:%S",
        run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> dict[str, Any]:
        """Get current time asynchronously.

        Args:
            format: Format string for datetime output
            run_manager: Callback manager for async tool execution

        Returns:
            Dict[str, Any]: Response containing current time information

        Raises:
            ValueError: If time formatting fails
        """
        logger.info("Getting current time asynchronously")
        try:
            formatted_time, timestamp = self._get_time(format)
            response = GetCurrentTimeResponse(
                current_time=formatted_time,
                timestamp=timestamp
            ).model_dump()
            logger.info(f"Successfully retrieved current time: {formatted_time}")
            return response
        except Exception as e:
            logger.exception("Failed to get current time")
            error_response = GetCurrentTimeResponse(
                current_time="",
                timestamp=0.0,
                error=str(e)
            ).model_dump()
            logger.error(f"Returning error response: {error_response}")
            return error_response


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Test the current time tool."""
        # Configure logger for testing
        # logger.add(
        #     "current_time.log",
        #     rotation="1 MB",
        #     retention="10 days",
        #     level="DEBUG"
        # )

        tool = GetCurrentTimeTool()
        logger.info("Starting current time tool tests")

        # Test default format
        logger.info("Testing default time format...")
        result = await tool.arun({})
        logger.info(f"Default format result: {result}")
        print(f"Default format result: {result}")

        # Test custom format
        logger.info("Testing custom time format...")
        result = await tool.arun({"format": "%H:%M:%S"})
        logger.info(f"Custom format result: {result}")
        print(f"Custom format result: {result}")

        # Test invalid format
        logger.info("Testing invalid time format...")
        result = await tool.arun({"format": "invalid"})
        logger.info(f"Invalid format result: {result}")
        print(f"Invalid format result: {result}")

        logger.info("Current time tool tests completed")

    asyncio.run(main())
