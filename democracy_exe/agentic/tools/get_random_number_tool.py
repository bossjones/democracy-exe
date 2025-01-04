# pyright: reportAttributeAccessIssue=false
"""Random number generation tool for LangChain/LangGraph integration."""
from __future__ import annotations

import random

from typing import Any, Dict, Optional, Type, Union

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from loguru import logger
from pydantic import BaseModel, Field, field_validator


class GetRandomNumberInput(BaseModel):
    """Schema for random number tool input.

    Attributes:
        min_value: Minimum value (inclusive, defaults to 1)
        max_value: Maximum value (inclusive, defaults to 100)
        seed: Optional random seed for reproducibility
    """
    min_value: int = Field(
        default=1,
        description="Minimum value (inclusive)"
    )
    max_value: int = Field(
        default=100,
        description="Maximum value (inclusive)"
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility"
    )

    @field_validator("max_value", mode="before")
    @classmethod
    def validate_max_value(cls, v: int, info: Any) -> int:
        """Validate that max_value is greater than min_value.

        Args:
            v: Maximum value to validate
            info: Validation context information

        Returns:
            int: Validated maximum value

        Raises:
            ValueError: If max_value is not greater than min_value
        """
        min_value = info.data.get("min_value")
        if min_value is not None and v <= min_value:
            raise ValueError(f"max_value ({v}) must be greater than min_value ({min_value})")
        return v


class GetRandomNumberResponse(BaseModel):
    """Schema for random number tool response.

    Attributes:
        random_number: Generated random number
        min_value: Minimum value used
        max_value: Maximum value used
        seed: Seed used (if any)
        error: Optional error message if operation failed
    """
    random_number: int = Field(description="Generated random number")
    min_value: int = Field(description="Minimum value used")
    max_value: int = Field(description="Maximum value used")
    seed: int | None = Field(
        default=None,
        description="Seed used (if any)"
    )
    error: str | None = Field(
        default=None,
        description="Error message if operation failed"
    )


class GetRandomNumberTool(BaseTool):
    """Tool for generating random numbers within a specified range.

    This tool provides random number generation with optional seed for reproducibility.
    It supports both synchronous and asynchronous operations.

    Attributes:
        name: Tool name for LangChain
        description: Tool description for LLM context
        args_schema: Pydantic model for argument validation
        return_direct: Whether to return results directly to user
    """
    name: str = "random_number"
    description: str = """Generate a random number within a specified range.
    Supports optional seed for reproducibility."""
    args_schema: type[BaseModel] = GetRandomNumberInput
    return_direct: bool = True

    def _generate_number(
        self,
        min_value: int = 1,
        max_value: int = 100,
        seed: int | None = None
    ) -> int:
        """Generate a random number within the specified range.

        Args:
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            seed: Optional random seed

        Returns:
            int: Generated random number

        Raises:
            ValueError: If range is invalid
        """
        if max_value <= min_value:
            raise ValueError("Maximum value must be greater than minimum value")

        try:
            if seed is not None:
                logger.debug("Generating random number with seed")
                random.seed(seed)
            else:
                logger.debug("Generating random number without seed")

            number = random.randint(min_value, max_value) # noqa: S311
            logger.debug(f"Generated number: {number}")
            return number
        except Exception as e:
            logger.error(f"Error generating random number: {e!s}")
            raise ValueError(f"Failed to generate random number: {e!s}")

    def _run(
        self,
        min_value: int = 1,
        max_value: int = 100,
        seed: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> dict[str, Any]:
        """Generate random number synchronously.

        Args:
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            seed: Optional random seed
            run_manager: Callback manager for tool execution

        Returns:
            Dict[str, Any]: Response containing generated number

        Raises:
            ValueError: If number generation fails
        """
        logger.info("Generating random number synchronously")
        try:
            number = self._generate_number(min_value, max_value, seed)
            response = GetRandomNumberResponse(
                random_number=number,
                min_value=min_value,
                max_value=max_value,
                seed=seed
            ).dict()
            logger.info(f"Successfully generated random number: {number}")
            return response
        except Exception as e:
            logger.exception("Failed to generate random number")
            error_response = GetRandomNumberResponse(
                random_number=0,
                min_value=min_value,
                max_value=max_value,
                seed=seed,
                error=str(e)
            ).dict()
            logger.error(f"Returning error response: {error_response}")
            return error_response

    async def _arun(
        self,
        min_value: int = 1,
        max_value: int = 100,
        seed: int | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> dict[str, Any]:
        """Generate random number asynchronously.

        Args:
            min_value: Minimum value (inclusive)
            max_value: Maximum value (inclusive)
            seed: Optional random seed
            run_manager: Callback manager for async tool execution

        Returns:
            Dict[str, Any]: Response containing generated number

        Raises:
            ValueError: If number generation fails
        """
        logger.info("Generating random number asynchronously")
        try:
            number = self._generate_number(min_value, max_value, seed)
            response = GetRandomNumberResponse(
                random_number=number,
                min_value=min_value,
                max_value=max_value,
                seed=seed
            ).dict()
            logger.info(f"Successfully generated random number: {number}")
            return response
        except Exception as e:
            logger.exception("Failed to generate random number")
            error_response = GetRandomNumberResponse(
                random_number=0,
                min_value=min_value,
                max_value=max_value,
                seed=seed,
                error=str(e)
            ).dict()
            logger.error(f"Returning error response: {error_response}")
            return error_response


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Test the random number tool."""
        # Configure logger for testing
        # logger.add(
        #     "random_number.log",
        #     rotation="1 MB",
        #     retention="10 days",
        #     level="DEBUG"
        # )

        tool = GetRandomNumberTool()
        logger.info("Starting random number tool tests")

        # Test default range
        logger.info("Testing default range...")
        result = await tool.arun({})
        logger.info(f"Default range result: {result}")
        print(f"Default range result: {result}")

        # Test custom range
        logger.info("Testing custom range...")
        result = await tool.arun({
            "min_value": 1000,
            "max_value": 2000
        })
        logger.info(f"Custom range result: {result}")
        print(f"Custom range result: {result}")

        # Test with seed
        logger.info("Testing with seed...")
        result = await tool.arun({
            "min_value": 1,
            "max_value": 100,
            "seed": 42
        })
        logger.info(f"Seeded result: {result}")
        print(f"Seeded result: {result}")

        # Test invalid range
        logger.info("Testing invalid range...")
        result = await tool.arun({
            "min_value": 100,
            "max_value": 1  # This will fail validation
        })
        logger.info(f"Invalid range result: {result}")
        print(f"Invalid range result: {result}")

        logger.info("Random number tool tests completed")

    asyncio.run(main())
