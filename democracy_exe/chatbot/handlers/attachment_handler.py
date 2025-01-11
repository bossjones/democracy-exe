# pyright: reportAttributeAccessIssue=false
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


logger = structlog.get_logger(__name__)
from PIL import Image


class AttachmentHandler:
    """Handles processing and saving of Discord attachments."""

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
            logger.error(f"Error converting attachment to dict: {e}")
            raise ValueError(f"Failed to convert attachment to dict: {e}") from e

    @staticmethod
    def file_to_local_data_dict(fname: str, dir_root: str) -> dict[str, Any]:
        """Convert a file to a dictionary with metadata.

        Args:
            fname: The name of the file to be converted
            dir_root: The root directory where the file is located

        Returns:
            A dictionary containing metadata about the file

        Raises:
            FileNotFoundError: If the file does not exist
            OSError: If there's an error accessing file stats
        """
        try:
            file_api = pathlib.Path(fname)
            if not file_api.exists():
                raise FileNotFoundError(f"File not found: {fname}")

            return {
                "filename": f"{dir_root}/{file_api.stem}{file_api.suffix}",
                "size": file_api.stat().st_size,
                "ext": f"{file_api.suffix}",
                "api": file_api,
            }
        except Exception as e:
            logger.error(f"Error creating file metadata dict: {e}")
            raise

    @staticmethod
    async def download_image(url: str) -> BytesIO | None:
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
        try:
            # Add timeout for download
            async with asyncio.timeout(30.0):
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            # Check content length before downloading
                            content_length = response.content_length
                            if content_length and content_length > 8 * 1024 * 1024:  # 8MB limit
                                raise RuntimeError(f"Image size {content_length} exceeds 8MB limit")

                            # Stream download with size limit
                            data = bytearray()
                            chunk_size = 8192  # 8KB chunks
                            total_size = 0

                            async for chunk in response.content.iter_chunked(chunk_size):
                                total_size += len(chunk)
                                if total_size > 8 * 1024 * 1024:  # 8MB limit
                                    raise RuntimeError("Image download exceeds 8MB limit")
                                data.extend(chunk)

                            return io.BytesIO(data)
                        else:
                            logger.error("Failed to download image", status=response.status)
                            return None
        except TimeoutError:
            logger.error("Image download timed out", url=url)
            raise
        except aiohttp.ClientError as e:
            logger.error("Error downloading image", error=str(e), url=url)
            raise
        except Exception as e:
            logger.error("Unexpected error downloading image", error=str(e), url=url)
            raise

    @staticmethod
    async def file_to_data_uri(file: File) -> str:
        """Convert a discord.File object to a data URI.

        Args:
            file: The discord.File object to be converted

        Returns:
            A data URI representing the file content

        Raises:
            ValueError: If file is not readable or exceeds size limits
            RuntimeError: If file size exceeds limits
        """
        try:
            if not file.fp or not file.fp.readable():
                raise ValueError("File is not readable")

            # Check file size before reading
            file.fp.seek(0, os.SEEK_END)
            file_size = file.fp.tell()
            file.fp.seek(0)

            if file_size > 8 * 1024 * 1024:  # 8MB limit
                raise RuntimeError(f"File size {file_size} exceeds 8MB limit")

            with BytesIO(file.fp.read()) as f:
                file_bytes = f.read()
            base64_encoded = base64.b64encode(file_bytes).decode("ascii")

            # Check encoded size
            if len(base64_encoded) > 10 * 1024 * 1024:  # 10MB limit for base64
                raise RuntimeError("Base64 encoded size exceeds limit")

            return f"data:image;base64,{base64_encoded}"
        except Exception as e:
            logger.error("Error converting file to data URI", error=str(e))
            raise

    @staticmethod
    async def data_uri_to_file(data_uri: str, filename: str) -> File:
        """Convert a data URI to a discord.File object.

        Args:
            data_uri: The data URI to be converted
            filename: The name of the file to be created

        Returns:
            A discord.File object containing the decoded data

        Raises:
            ValueError: If data URI is invalid or exceeds size limits
            RuntimeError: If decoded data exceeds size limits
        """
        try:
            if "," not in data_uri:
                raise ValueError("Invalid data URI format")

            # Check data URI size
            if len(data_uri) > 10 * 1024 * 1024:  # 10MB limit
                raise RuntimeError("Data URI size exceeds limit")

            metadata, base64_data = data_uri.split(",")

            # Check base64 data size
            if len(base64_data) > 8 * 1024 * 1024:  # 8MB limit
                raise RuntimeError("Base64 data size exceeds limit")

            file_bytes = base64.b64decode(base64_data)

            # Check decoded size
            if len(file_bytes) > 8 * 1024 * 1024:  # 8MB limit
                raise RuntimeError("Decoded file size exceeds limit")

            return discord.File(BytesIO(file_bytes), filename=filename, spoiler=False)
        except Exception as e:
            logger.error("Error converting data URI to file", error=str(e))
            raise

    @staticmethod
    def path_for(attm: Attachment, basedir: str = "./") -> pathlib.Path:
        """Generate a pathlib.Path object for an attachment.

        Args:
            attm: The attachment for which the path is generated
            basedir: The base directory path. Default is current directory

        Returns:
            A pathlib.Path object representing the path for the attachment file

        Raises:
            ValueError: If attachment filename is invalid
        """
        try:
            if not attm.filename:
                raise ValueError("Attachment has no filename")

            p = pathlib.Path(basedir).resolve() / str(attm.filename)
            logger.debug(f"path_for: p -> {p}")
            return p
        except Exception as e:
            logger.error(f"Error generating path for attachment: {e}")
            raise

    async def save_attachment(self, attm: Attachment, basedir: str = "./") -> None:
        """Save a Discord attachment to a specified directory.

        Args:
            attm: The attachment to be saved
            basedir: The base directory path where the file will be saved

        Raises:
            HTTPException: If there's an error saving the attachment
            OSError: If there's an error creating directories
            RuntimeError: If attachment size exceeds limits
            ValueError: If file type is not allowed
        """
        try:
            # Check attachment size
            if attm.size > 8 * 1024 * 1024:  # 8MB limit
                raise RuntimeError(f"Attachment size {attm.size} exceeds 8MB limit")

            # Verify file type
            allowed_types = {
                'image/jpeg', 'image/png', 'image/gif', 'image/webp',
                'text/plain', 'application/json', 'text/markdown'
            }

            if attm.content_type not in allowed_types:
                raise ValueError(f"File type {attm.content_type} not allowed. Allowed types: {allowed_types}")

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

            # Verify path is within basedir to prevent directory traversal
            basedir_path = pathlib.Path(basedir).resolve()
            file_path = path.resolve()
            if not str(file_path).startswith(str(basedir_path)):
                raise ValueError("Invalid file path - potential directory traversal attempt")

            path.parent.mkdir(parents=True, exist_ok=True)

            # Add timeout for save operation
            async with asyncio.timeout(30.0):
                try:
                    await attm.save(path, use_cached=True)
                    await asyncio.sleep(5)
                except HTTPException:
                    await attm.save(path)

                # Verify saved file
                if not path.exists():
                    raise RuntimeError("File was not saved successfully")
                if path.stat().st_size != attm.size:
                    path.unlink()  # Delete corrupted file
                    raise RuntimeError("Saved file size does not match attachment size")

        except TimeoutError:
            logger.error("Attachment save timed out", path=str(path))
            if path.exists():
                path.unlink()  # Cleanup partial file
            raise
        except Exception as e:
            logger.error("Error saving attachment", error=str(e), path=str(path))
            if 'path' in locals() and path.exists():
                path.unlink()  # Cleanup on error
            raise

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
        try:
            if not all(key in attm_data_dict for key in ["id", "filename", "attachment_obj"]):
                raise ValueError("Invalid attachment data dictionary")

            # Check attachment size
            if attm_data_dict.get("size", 0) > 8 * 1024 * 1024:  # 8MB limit
                raise RuntimeError(f"Attachment size {attm_data_dict['size']} exceeds 8MB limit")

            fname = f"{dir_root}/orig_{attm_data_dict['id']}_{attm_data_dict['filename']}"
            rich.print(f"Saving to ... {fname}")

            await attm_data_dict["attachment_obj"].save(fname, use_cached=True)
            await asyncio.sleep(1)
            return fname

        except Exception as e:
            logger.error(f"Error saving attachment locally: {e}")
            raise

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
