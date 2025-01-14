# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
"""File deletion tool for LangChain/LangGraph integration."""
from __future__ import annotations

import os
import pathlib

from typing import Any, Dict, Optional, Type, Union

import structlog

from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.tools import BaseTool


logger = structlog.get_logger(__name__)
from pydantic import BaseModel, Field

from democracy_exe.utils.file_functions import fix_path, is_file


class DeleteFileInput(BaseModel):
    """Schema for file deletion tool input.

    Attributes:
        file_name: Name of the file to delete
        directory: Optional target directory (defaults to ./scratchpad)
        force: Whether to force deletion without confirmation
    """
    file_name: str = Field(description="Name of the file to delete")
    directory: str | None = Field(
        default="./scratchpad",
        description="Target directory containing the file"
    )
    force: bool = Field(
        default=False,
        description="Whether to force deletion without confirmation"
    )


class DeleteFileResponse(BaseModel):
    """Schema for file deletion tool response.

    Attributes:
        file_path: Full path to the deleted file
        status: Status of the file deletion operation
        error: Optional error message if operation failed
        requires_confirmation: Whether confirmation is required before deletion
    """
    file_path: str = Field(description="Full path to the deleted file")
    status: str = Field(description="Status of the file deletion operation")
    error: str | None = Field(
        default=None,
        description="Error message if operation failed"
    )
    requires_confirmation: bool = Field(
        default=False,
        description="Whether confirmation is required before deletion"
    )


class DeleteFileTool(BaseTool):
    """Tool for deleting files with proper validation and confirmation.

    This tool handles file deletion operations with proper validation and error handling.
    It supports both synchronous and asynchronous operations and includes a confirmation
    mechanism for safe deletions.

    Attributes:
        name: Tool name for LangChain
        description: Tool description for LLM context
        args_schema: Pydantic model for argument validation
        return_direct: Whether to return results directly to user
    """
    name: str = "file_deleter"
    description: str = """Delete a file from a given directory with optional force flag.
    Handles file deletion with proper validation and confirmation checks."""
    args_schema: type[BaseModel] = DeleteFileInput
    return_direct: bool = True

    def _validate_path(self, directory: str, file_name: str) -> tuple[pathlib.Path, str]:
        """Validate and prepare the file path.

        Args:
            directory: Target directory path
            file_name: Name of the file to delete

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
        directory: str = "./scratchpad",
        force: bool = False,
        run_manager: CallbackManagerForToolRun | None = None
    ) -> dict[str, Any]:
        """Delete a file synchronously.

        Args:
            file_name: Name of the file to delete
            directory: Target directory (defaults to ./scratchpad)
            force: Whether to force deletion without confirmation
            run_manager: Callback manager for tool execution

        Returns:
            Dict[str, Any]: Response containing file deletion status

        Raises:
            ValueError: If file deletion fails
        """
        logger.info(f"Starting synchronous file deletion for {file_name} in {directory}")
        try:
            # Validate paths
            logger.debug("Validating paths...")
            dir_path, file_path = self._validate_path(directory, file_name)
            logger.debug(f"Path validation successful. Directory: {dir_path}, File: {file_path}")

            # Check if confirmation is required
            if not force:
                logger.info("Force flag not set, requiring confirmation")
                return DeleteFileResponse(
                    file_path=file_path,
                    status="confirmation_required",
                    requires_confirmation=True
                ).dict()

            # Delete the file
            logger.debug(f"Deleting file: {file_path}")
            os.remove(file_path)
            logger.info(f"Successfully deleted file: {file_path}")

            response = DeleteFileResponse(
                file_path=file_path,
                status="success"
            ).model_dump()
            logger.info("File deletion completed successfully")
            return response
        except Exception as e:
            logger.exception(f"File deletion failed: {e!s}")
            error_response = DeleteFileResponse(
                file_path="",
                status="error",
                error=str(e)
            ).dict()
            logger.error(f"Returning error response: {error_response}")
            return error_response

    async def _arun(
        self,
        file_name: str,
        directory: str = "./scratchpad",
        force: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None
    ) -> dict[str, Any]:
        """Delete a file asynchronously.

        Args:
            file_name: Name of the file to delete
            directory: Target directory (defaults to ./scratchpad)
            force: Whether to force deletion without confirmation
            run_manager: Callback manager for async tool execution

        Returns:
            Dict[str, Any]: Response containing file deletion status

        Raises:
            ValueError: If file deletion fails
        """
        logger.info(f"Starting asynchronous file deletion for {file_name} in {directory}")
        try:
            # Validate paths
            logger.debug("Validating paths...")
            dir_path, file_path = self._validate_path(directory, file_name)
            logger.debug(f"Path validation successful. Directory: {dir_path}, File: {file_path}")

            # Check if confirmation is required
            if not force:
                logger.info("Force flag not set, requiring confirmation")
                return DeleteFileResponse(
                    file_path=file_path,
                    status="confirmation_required",
                    requires_confirmation=True
                ).dict()

            # Delete the file
            logger.debug(f"Deleting file: {file_path}")
            os.remove(file_path)  # os.remove is fast enough to not need async
            logger.info(f"Successfully deleted file: {file_path}")

            response = DeleteFileResponse(
                file_path=file_path,
                status="success"
            ).dict()
            logger.info("Asynchronous file deletion completed successfully")
            return response
        except Exception as e:
            logger.exception(f"Asynchronous file deletion failed: {e!s}")
            error_response = DeleteFileResponse(
                file_path="",
                status="error",
                error=str(e)
            ).dict()
            logger.error(f"Returning error response: {error_response}")
            return error_response


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        """Test the file deletion tool."""
        # Configure logger for testing
        # logger.add(
        #     "file_deletion.log",
        #     rotation="1 MB",
        #     retention="10 days",
        #     level="DEBUG"
        # )

        tool = DeleteFileTool()
        logger.info("Starting file deletion tool tests")

        # Create a test file first
        test_dir = pathlib.Path("./scratchpad")
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / "test.txt"
        test_file.write_text("Test content")

        # Test deletion without force flag
        logger.info("Testing deletion without force flag...")
        result = await tool.arun({
            "file_name": "test.txt",
            "directory": "./scratchpad",
            "force": False
        })
        logger.info(f"No-force Result: {result}")
        print(f"No-force Result: {result}")

        # Test forced deletion
        logger.info("Testing forced deletion...")
        result = await tool.arun({
            "file_name": "test.txt",
            "directory": "./scratchpad",
            "force": True
        })
        logger.info(f"Force Result: {result}")
        print(f"Force Result: {result}")

        logger.info("File deletion tool tests completed")

    asyncio.run(main())
