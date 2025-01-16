# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""Attachment handling functionality.

This module contains functionality for handling Discord attachments,
including saving, downloading, and processing attachments.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import pathlib
import uuid

from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import aiohttp
import discord
import rich
import structlog

from discord import Attachment, File, HTTPException, Message
from PIL import Image

from democracy_exe.aio_settings import aiosettings
from democracy_exe.chatbot.utils.resource_manager import ResourceLimits, ResourceManager
from democracy_exe.constants import MAX_BYTES_UPLOAD_DISCORD, MAX_FILE_UPLOAD_IMAGES_IMGUR


logger = structlog.get_logger(__name__)


class AttachmentHandler:
    """Handles processing and saving of Discord attachments."""

    def __init__(self) -> None:
        """Initialize the attachment handler."""
        self._download_semaphore = asyncio.Semaphore(
            getattr(aiosettings, "max_concurrent_downloads", 5)
        )
        limits = ResourceLimits(
            max_memory_mb=getattr(aiosettings, "max_memory_mb", 512),
            max_tasks=getattr(aiosettings, "max_tasks", 100),
            max_response_size_mb=getattr(aiosettings, "max_response_size_mb", 1),
            max_buffer_size_kb=getattr(aiosettings, "max_buffer_size_kb", 64),
            task_timeout_seconds=getattr(aiosettings, "task_timeout_seconds", 30.0)
        )
        self._resource_manager = ResourceManager(limits)
        self._max_total_size = MAX_BYTES_UPLOAD_DISCORD
        self._max_image_size = MAX_FILE_UPLOAD_IMAGES_IMGUR

    @staticmethod
    def attachment_to_dict(attm: Attachment) -> dict[str, Any]:
        """Convert a discord.Attachment object to a dictionary.

        Args:
            attm: The attachment object to be converted

        Returns:
            A dictionary containing information about the attachment

        Raises:
            ValueError: If attachment is missing required attributes
        """
        try:
            if not all(hasattr(attm, attr) for attr in ["filename", "id", "url"]):
                raise ValueError("Attachment missing required attributes")

            result = {
                "filename": attm.filename,
                "id": attm.id,
                "proxy_url": attm.proxy_url,
                "size": attm.size,
                "url": attm.url,
                "spoiler": attm.is_spoiler(),
            }

            # Optional attributes
            if attm.height is not None:
                result["height"] = attm.height
            if attm.width is not None:
                result["width"] = attm.width
            if attm.content_type is not None:
                result["content_type"] = attm.content_type

            result["attachment_obj"] = attm
            return result

        except Exception as e:
            logger.error("Error converting attachment to dict", error=str(e))
            raise ValueError(f"Failed to convert attachment to dict: {e}") from e

    async def download_image(self, url: str) -> BytesIO | None:
        """Download an image from a given URL asynchronously.

        Args:
            url: The URL of the image to download

        Returns:
            A BytesIO object containing the downloaded image data, or None if download fails

        Raises:
            aiohttp.ClientError: If there's an error downloading the image
            RuntimeError: If image size exceeds limits
            asyncio.TimeoutError: If download times out
        """
        task = asyncio.current_task()
        if task:
            await self._resource_manager.track_task(task)

        try:
            timeout = self._resource_manager.limits.task_timeout_seconds
            max_size = self._max_image_size

            # Add timeout for download
            async with asyncio.timeout(timeout):
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            # Check content length before downloading
                            content_length = response.content_length
                            if content_length and content_length > max_size:
                                raise RuntimeError(f"Image size {content_length} exceeds {max_size} limit")

                            # Stream download with size limit
                            data = bytearray()
                            chunk_size = self._resource_manager.limits.max_buffer_size_kb * 1024
                            total_size = 0

                            async for chunk in response.content.iter_chunked(chunk_size):
                                total_size += len(chunk)
                                if total_size > max_size:
                                    raise RuntimeError(f"Image download exceeds {max_size} limit")

                                # Track memory for chunk
                                self._resource_manager.track_memory(len(chunk))
                                data.extend(chunk)
                                self._resource_manager.release_memory(len(chunk))

                            return io.BytesIO(data)
                        else:
                            logger.error("Failed to download image", status=response.status)
                            return None

        except TimeoutError:
            logger.error("Image download timed out", url=url)
            raise RuntimeError(f"Image download timed out after {timeout} seconds")

        except aiohttp.ClientError as e:
            logger.error("Error downloading image", error=str(e), url=url)
            raise

        except Exception as e:
            logger.error("Unexpected error downloading image", error=str(e), url=url)
            raise

        finally:
            if task:
                await self._resource_manager.cleanup_tasks([task])
                logger.info("Resource cleanup completed", task=str(task))

    async def save_attachment(self, attm: Attachment, basedir: str = "./") -> None:
        """Save a Discord attachment to a specified directory.

        Args:
            attm: The attachment to be saved
            basedir: The base directory path where the file will be saved

        Raises:
            HTTPException: If there's an error saving the attachment
            OSError: If there's an error creating directories
            RuntimeError: If attachment size exceeds limits
            ValueError: If file type is not allowed or path is unsafe
        """
        task = asyncio.current_task()
        if task:
            await self._resource_manager.track_task(task)

        path = None
        try:
            # Check attachment size
            if attm.size > self._max_total_size:
                raise RuntimeError(f"Attachment size {attm.size} exceeds {self._max_total_size} limit")

            # Track memory for attachment
            self._resource_manager.track_memory(attm.size)

            # Verify file type
            allowed_types = {
                'image/jpeg', 'image/png', 'image/gif', 'image/webp',
                'text/plain', 'application/json', 'text/markdown'
            }

            if attm.content_type not in allowed_types:
                raise ValueError(f"File type {attm.content_type} not allowed. Allowed types: {allowed_types}")

            # Get safe path (this will raise ValueError for directory traversal)
            path = self.path_for(attm, basedir=basedir)
            logger.debug("save_attachment: path", path=str(path))

            # Check available disk space
            try:
                disk_usage = os.statvfs(path.parent)
                available_space = disk_usage.f_frsize * disk_usage.f_bavail
                if available_space < attm.size * 2:  # Require 2x space
                    raise RuntimeError("Insufficient disk space")
            except AttributeError:
                # statvfs not available on Windows
                pass

            path.parent.mkdir(parents=True, exist_ok=True)

            # Add timeout for save operation
            timeout = self._resource_manager.limits.task_timeout_seconds
            async with asyncio.timeout(timeout):
                try:
                    await attm.save(str(path), use_cached=True)
                    await asyncio.sleep(0.1)  # Small delay to ensure file is written
                except HTTPException:
                    await attm.save(str(path))

                # Verify saved file
                if not path.exists():
                    raise RuntimeError("File was not saved successfully")

                actual_size = path.stat().st_size
                if actual_size != attm.size:
                    path.unlink()  # Delete corrupted file
                    raise RuntimeError(f"Saved file size ({actual_size}) does not match attachment size ({attm.size})")

        except Exception as e:
            logger.error("Error saving attachment", error=str(e), path=str(path) if path else None)
            if path and path.exists():
                path.unlink()  # Cleanup on error
            raise

        finally:
            if task:
                await self._resource_manager.cleanup_tasks([task])
                logger.info("Resource cleanup completed", task=str(task))
            # Release memory for attachment
            self._resource_manager.release_memory(attm.size)

    def path_for(self, attm: Attachment, basedir: str = "./") -> pathlib.Path:
        """Generate a safe path for saving an attachment.

        Args:
            attm: The attachment to generate a path for
            basedir: The base directory path

        Returns:
            A Path object representing the safe file path

        Raises:
            ValueError: If the file path would result in directory traversal
        """
        # Check for directory traversal in original filename
        if ".." in attm.filename or attm.filename.startswith("/"):
            raise ValueError("Invalid file path - potential directory traversal attempt")

        # Clean filename to prevent directory traversal
        safe_filename = pathlib.Path(attm.filename).name
        base_path = pathlib.Path(basedir).resolve()
        file_path = (base_path / safe_filename).resolve()

        # Double-check that the resolved path is within the base directory
        try:
            file_path.relative_to(base_path)
        except ValueError:
            raise ValueError("Invalid file path - potential directory traversal attempt")

        return file_path

    async def handle_save_attachment_locally(self, attm_data_dict: dict[str, Any], dir_root: str) -> str:
        """Save a Discord attachment locally.

        Args:
            attm_data_dict: A dictionary containing information about the attachment
            dir_root: The root directory where the file will be saved

        Returns:
            The path of the saved attachment file

        Raises:
            ValueError: If attachment data is invalid
            HTTPException: If there's an error saving the attachment
            RuntimeError: If attachment size exceeds limits
        """
        task = asyncio.current_task()
        if task:
            await self._resource_manager.track_task(task)

        try:
            if not all(key in attm_data_dict for key in ["id", "filename", "attachment_obj"]):
                raise ValueError("Invalid attachment data dictionary")

            # Check attachment size
            if attm_data_dict.get("size", 0) > self._max_total_size:
                raise RuntimeError(f"Attachment size {attm_data_dict['size']} exceeds {self._max_total_size} limit")

            # Track memory for attachment
            size = attm_data_dict.get("size", 0)
            self._resource_manager.track_memory(size)

            fname = f"{dir_root}/orig_{attm_data_dict['id']}_{attm_data_dict['filename']}"
            logger.debug("Saving attachment locally", path=fname)

            timeout = self._resource_manager.limits.task_timeout_seconds
            async with asyncio.timeout(timeout):
                await attm_data_dict["attachment_obj"].save(fname, use_cached=True)
                await asyncio.sleep(0.1)  # Small delay to ensure file is written

            return fname

        except Exception as e:
            logger.error("Error saving attachment locally", error=str(e))
            raise

        finally:
            if task:
                await self._resource_manager.cleanup_tasks([task])
                logger.info("Resource cleanup completed", task=str(task))
            # Release memory for attachment
            size = attm_data_dict.get("size", 0)
            self._resource_manager.release_memory(size)

    def get_attachments(
        self, message: Message
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]], list[str]]:
        """Retrieve attachment data from a Discord message.

        Args:
            message: The Discord message containing attachments

        Returns:
            A tuple containing:
                - A list of dictionaries with attachment data
                - A list of local attachment file paths
                - A list of dictionaries with local attachment data
                - A list of media filepaths

        Raises:
            ValueError: If message has no attachments
        """
        try:
            if not message.attachments:
                logger.warning("Message has no attachments")
                return [], [], [], []

            attachment_data_list_dicts = []
            local_attachment_file_list = []
            local_attachment_data_list_dicts = []
            media_filepaths = []

            for attm in message.attachments:
                try:
                    data = self.attachment_to_dict(attm)
                    attachment_data_list_dicts.append(data)
                except Exception as e:
                    logger.error(f"Error processing attachment: {e}")
                    continue

            return (
                attachment_data_list_dicts,
                local_attachment_file_list,
                local_attachment_data_list_dicts,
                media_filepaths,
            )

        except Exception as e:
            logger.error(f"Error getting attachments: {e}")
            return [], [], [], []
