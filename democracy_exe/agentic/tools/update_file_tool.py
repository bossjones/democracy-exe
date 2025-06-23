# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
"""File update tool for LangChain/LangGraph integration."""
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


class UpdateFileInput(BaseModel):
    """Schema for file update tool input.

    Attributes:
        file_name: Name of the file to update
        content: New content to write to the file
        directory: Optional target directory (defaults to ./scratchpad)
    """
    file_name: str = Field(description="Name of the file to update")
    content: str = Field(description="New content to write to the file")
    directory: str | None = Field(
        default="./scratchpad",
        description="Target directory containing the file"
    )


class UpdateFileResponse(BaseModel):
    """Schema for file update tool response.

    Attributes:
        file_path: Full path to the updated file
        status: Status of the file update operation
        error: Optional error message if operation failed
    """
    file_path: str = Field(description="Full path to the updated file")
    status: str = Field(description="Status of the file update operation")
    error: str | None = Field(
        default=None,
        description="Error message if operation failed"
    )


class UpdateFileTool(BaseTool):
    """Tool for updating existing files with new content.

    This tool handles file update operations with proper validation and error handling.
    It supports both synchronous and asynchronous operations using aiofiles.

    Attributes:
        name: Tool name for LangChain
        description: Tool description for LLM context
        args_schema: Pydantic model for argument validation
        return_direct: Whether to return results directly to user
    """
    name: str = "file_updater"
    description: str = """Update an existing file with new content in a given directory.
    Handles file updates with proper validation and error handling."""
    args_schema: type[BaseModel] = UpdateFileInput
    return_direct: bool = True

    def _validate_path(self, directory: str, file_name: str) -> tuple[pathlib.Path, str]:
        """Validate and prepare the file path.

        Args:
            directory: Target directory path
            file_name: Name of the file to update

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
                raise ValueError(f"Directory does not exist: {dir_path}")

            # Validate file exists
            if not is_file(str(file_path)):
                raise ValueError(f"File does not exist: {file_path}")

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
        """Update a file synchronously.

        Args:
            file_name: Name of the file to update
            content: New content to write to the file
            directory: Target directory (defaults to ./scratchpad)
            run_manager: Callback manager for tool execution

        Returns:
            Dict[str, Any]: Response containing file update status

        Raises:
            ValueError: If file update fails
        """
        logger.info(f"Starting synchronous file update for {file_name} in {directory}")
        try:
            # Validate paths
            logger.debug("Validating paths...")
            dir_path, file_path = self._validate_path(directory, file_name)
            logger.debug(f"Path validation successful. Directory: {dir_path}, File: {file_path}")

            # Read existing content for backup/logging
            logger.debug(f"Reading existing content from: {file_path}")
            with open(file_path, encoding='utf-8') as f:
                old_content = f.read()
            logger.debug(f"Original content length: {len(old_content)} characters")

            # Write new content
            logger.debug(f"Writing new content to file: {file_path}")
            content_length = len(content)
            logger.debug(f"New content length: {content_length} characters")

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Successfully wrote {content_length} characters to {file_path}")

            response = UpdateFileResponse(
                file_path=file_path,
                status="success"
            ).model_dump()
            logger.info("File update completed successfully")
            return response
        except Exception as e:
            logger.exception(f"File update failed: {e!s}")
            error_response = UpdateFileResponse(
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
        """Update a file asynchronously.

        Args:
            file_name: Name of the file to update
            content: New content to write to the file
            directory: Target directory (defaults to ./scratchpad)
            run_manager: Callback manager for async tool execution

        Returns:
            Dict[str, Any]: Response containing file update status

        Raises:
            ValueError: If file update fails
        """
        logger.info(f"Starting asynchronous file update for {file_name} in {directory}")
        try:
            # Validate paths
            logger.debug("Validating paths...")
            dir_path, file_path = self._validate_path(directory, file_name)
            logger.debug(f"Path validation successful. Directory: {dir_path}, File: {file_path}")

            # Read existing content for backup/logging
            logger.debug(f"Reading existing content from: {file_path}")
            async with aiofiles.open(file_path, encoding='utf-8') as f:
                old_content = await f.read()
            logger.debug(f"Original content length: {len(old_content)} characters")

            # Write new content asynchronously
            logger.debug(f"Writing new content to file asynchronously: {file_path}")
            content_length = len(content)
            logger.debug(f"New content length: {content_length} characters")

            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            logger.info(f"Successfully wrote {content_length} characters to {file_path}")

            response = UpdateFileResponse(
                file_path=file_path,
                status="success"
            ).model_dump()
            logger.info("Asynchronous file update completed successfully")
            return response
        except Exception as e:
            logger.exception(f"Asynchronous file update failed: {e!s}")
            error_response = UpdateFileResponse(
                file_path="",
                status="error",
                error=str(e)
            ).model_dump()
            logger.error(f"Returning error response: {error_response}")
            return error_response


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Test the file update tool."""
        # Configure logger for testing
        logger.add(
            "file_update.log",
            rotation="1 MB",
            retention="10 days",
            level="DEBUG"
        )

        tool = UpdateFileTool()
        logger.info("Starting file update tool tests")

        # Create a test file first
        test_dir = pathlib.Path("./scratchpad")
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.txt"
        test_file.write_text("Original content")

        # Test async file update
        logger.info("Testing async file update...")
        result = await tool.arun({
            "file_name": "test.txt",
            "content": "Updated content!",
            "directory": "./scratchpad"
        })
        logger.info(f"Async Result: {result}")
        print(f"Async Result: {result}")

        # Test sync file update
        logger.info("Testing sync file update...")
        result = tool.run({
            "file_name": "test.txt",
            "content": "Updated again!",
            "directory": "./scratchpad"
        })
        logger.info(f"Sync Result: {result}")
        print(f"Sync Result: {result}")

        logger.info("File update tool tests completed")

    asyncio.run(main())
