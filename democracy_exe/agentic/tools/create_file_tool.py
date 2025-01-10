# pyright: reportAttributeAccessIssue=false
"""File creation tool for LangChain/LangGraph integration."""
from __future__ import annotations

import os
import pathlib

from typing import Any, Dict, Optional, Type, Union

import aiofiles
import structlog

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool


logger = structlog.get_logger(__name__)
from pydantic import BaseModel, Field

from democracy_exe.utils.file_functions import fix_path, is_file


class CreateFileInput(BaseModel):
    """Schema for file creation tool input.

    Attributes:
        file_name: Name of the file to create
        content: Content to write to the file
        directory: Optional target directory (defaults to ./scratchpad)
    """
    file_name: str = Field(description="Name of the file to create")
    content: str = Field(description="Content to write to the file")
    directory: str | None = Field(
        default="./scratchpad",
        description="Target directory for file creation"
    )


class CreateFileResponse(BaseModel):
    """Schema for file creation tool response.

    Attributes:
        file_path: Full path to the created file
        status: Status of the file creation operation
        error: Optional error message if operation failed
    """
    file_path: str = Field(description="Full path to the created file")
    status: str = Field(description="Status of the file creation operation")
    error: str | None = Field(
        default=None,
        description="Error message if operation failed"
    )


class CreateFileTool(BaseTool):
    """Tool for creating files with specified content.

    This tool handles file creation operations with proper validation and error handling.
    It supports both synchronous and asynchronous operations using aiofiles.

    Attributes:
        name: Tool name for LangChain
        description: Tool description for LLM context
        args_schema: Pydantic model for argument validation
        return_direct: Whether to return results directly to user
    """
    name: str = "file_creator"
    description: str = """Create a new file with specified content in a given directory.
    Handles file creation with proper validation and error handling."""
    args_schema: type[BaseModel] = CreateFileInput
    return_direct: bool = True

    def _validate_path(self, directory: str, file_name: str) -> tuple[pathlib.Path, str]:
        """Validate and prepare the file path.

        Args:
            directory: Target directory path
            file_name: Name of the file to create

        Returns:
            tuple[pathlib.Path, str]: Tuple of (directory path, full file path)

        Raises:
            ValueError: If path validation fails
        """
        try:
            # Normalize directory path
            dir_path = pathlib.Path(fix_path(directory)).resolve()
            logger.debug(f"Normalized directory path: {dir_path}")

            # Create full file path
            file_path = dir_path / file_name
            logger.debug(f"Full file path: {file_path}")

            # Validate directory exists
            if not dir_path.exists():
                logger.warning(f"Directory does not exist: {dir_path}")
                raise ValueError(f"Directory does not exist: {dir_path}")

            # Validate file doesn't exist
            if is_file(str(file_path)):
                logger.warning(f"File already exists: {file_path}")
                raise ValueError(f"File already exists: {file_path}")

            return dir_path, str(file_path)
        except Exception as e:
            logger.error(f"Path validation failed: {e!s}")
            raise ValueError(f"Path validation failed: {e!s}")

    def _run(
        self,
        file_name: str,
        content: str,
        directory: str = "./scratchpad",
        run_manager: CallbackManagerForToolRun | None = None
    ) -> dict[str, Any]:
        """Create a file synchronously.

        Args:
            file_name: Name of the file to create
            content: Content to write to the file
            directory: Target directory (defaults to ./scratchpad)
            run_manager: Callback manager for tool execution

        Returns:
            Dict[str, Any]: Response containing file creation status

        Raises:
            ValueError: If file creation fails
        """
        logger.info(f"Starting synchronous file creation for {file_name} in {directory}")
        try:
            # Ensure directory exists
            dir_path = pathlib.Path(fix_path(directory)).resolve()
            logger.debug(f"Creating directory if not exists: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            logger.debug("Directory creation complete")

            # Validate paths
            logger.debug("Validating paths...")
            dir_path, file_path = self._validate_path(str(dir_path), file_name)
            logger.debug(f"Path validation successful. Directory: {dir_path}, File: {file_path}")

            # Write file content
            logger.debug(f"Writing content to file: {file_path}")
            content_length = len(content)
            logger.debug(f"Content length: {content_length} characters")

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Successfully wrote {content_length} characters to {file_path}")

            response = CreateFileResponse(
                file_path=file_path,
                status="success"
            ).model_dump()
            logger.info("Synchronous file creation completed successfully")
            return response
        except Exception as e:
            logger.exception(f"File creation failed: {e!s}")
            error_response = CreateFileResponse(
                file_path="",
                status="error",
                error=str(e)
            ).model_dump()
            logger.error(f"Returning error response: {error_response}")
            return error_response

    async def _arun(
        self,
        file_name: str,
        content: str,
        directory: str = "./scratchpad",
        run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> dict[str, Any]:
        """Create a file asynchronously.

        Args:
            file_name: Name of the file to create
            content: Content to write to the file
            directory: Target directory (defaults to ./scratchpad)
            run_manager: Callback manager for async tool execution

        Returns:
            Dict[str, Any]: Response containing file creation status

        Raises:
            ValueError: If file creation fails
        """
        logger.info(f"Starting asynchronous file creation for {file_name} in {directory}")
        try:
            # Validate paths
            logger.debug("Validating paths...")
            dir_path, file_path = self._validate_path(directory, file_name)
            logger.debug(f"Path validation successful. Directory: {dir_path}, File: {file_path}")

            # Ensure directory exists
            logger.debug(f"Creating directory if not exists: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
            logger.debug("Directory creation/verification complete")

            # Write file content asynchronously
            logger.debug(f"Writing content to file asynchronously: {file_path}")
            content_length = len(content)
            logger.debug(f"Content length: {content_length} characters")

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            logger.info(f"Successfully wrote {content_length} characters to {file_path}")

            response = CreateFileResponse(
                file_path=file_path,
                status="success"
            ).model_dump()
            logger.info("File creation completed successfully")
            return response
        except Exception as e:
            logger.exception(f"Asynchronous file creation failed: {e!s}")
            error_response = CreateFileResponse(
                file_path="",
                status="error",
                error=str(e)
            ).dict()
            logger.error(f"Returning error response: {error_response}")
            return error_response


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Test the file creation tool."""
        # Configure logger for testing
        # logger.add(
        #     "file_creation.log",
        #     rotation="1 MB",
        #     retention="10 days",
        #     level="DEBUG"
        # )

        tool = CreateFileTool()
        logger.info("Starting file creation tool tests")

        # Test async file creation
        logger.info("Testing async file creation...")
        result = await tool.arun({
            "file_name": "test.txt",
            "content": "Hello, World!",
            "directory": "./scratchpad"
        })
        logger.info(f"Async Result: {result}")
        print(f"Async Result: {result}")

        # Test sync file creation
        logger.info("Testing sync file creation...")
        result = tool.run({
            "file_name": "test2.txt",
            "content": "Hello again!",
            "directory": "./scratchpad"
        })
        logger.info(f"Sync Result: {result}")
        print(f"Sync Result: {result}")

        logger.info("File creation tool tests completed")

    asyncio.run(main())
