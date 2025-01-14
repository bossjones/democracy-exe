"""Resource manager for handling system resources and memory."""
from __future__ import annotations

import asyncio
import gc
import os

from dataclasses import dataclass
from typing import List, Optional, Set

import psutil
import structlog


logger = structlog.get_logger(__name__)

@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    max_memory_mb: int
    max_tasks: int
    max_response_size_mb: int
    max_buffer_size_kb: int
    task_timeout_seconds: float

class ResourceManager:
    """Manages system resources and memory."""

    def __init__(self, limits: ResourceLimits):
        """Initialize resource manager.

        Args:
            limits: Resource limits configuration
        """
        self._limits = limits
        self._tasks: set[asyncio.Task] = set()
        self._current_memory = 0

    async def check_memory(self) -> None:
        """Check if memory usage is within limits.

        Raises:
            RuntimeError: If memory usage exceeds limit
        """
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        if memory_mb > self._limits.max_memory_mb:
            await self.force_cleanup()
            raise RuntimeError(f"Memory usage {memory_mb:.2f}MB exceeds {self._limits.max_memory_mb}MB limit")

    def track_task(self, task: asyncio.Task) -> None:
        """Track a new task.

        Args:
            task: Task to track

        Raises:
            RuntimeError: If task limit is exceeded
        """
        if len(self._tasks) >= self._limits.max_tasks:
            raise RuntimeError(f"Task limit {self._limits.max_tasks} exceeded")
        self._tasks.add(task)

    async def cleanup_tasks(self, tasks: list[asyncio.Task] | None = None) -> None:
        """Clean up specified tasks or all tracked tasks.

        Args:
            tasks: Optional list of tasks to clean up. If None, cleans up all tracked tasks.
        """
        tasks_to_cleanup = tasks if tasks is not None else list(self._tasks)
        for task in tasks_to_cleanup:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            if task in self._tasks:
                self._tasks.remove(task)

    def track_memory(self, size_bytes: int) -> None:
        """Track memory allocation.

        Args:
            size_bytes: Size in bytes to track

        Raises:
            RuntimeError: If memory limit would be exceeded
        """
        new_total = self._current_memory + size_bytes
        if new_total > self._limits.max_memory_mb * 1024 * 1024:
            raise RuntimeError(f"Memory allocation of {size_bytes} bytes would exceed {self._limits.max_memory_mb}MB limit")
        self._current_memory = new_total

    def release_memory(self, size_bytes: int) -> None:
        """Release tracked memory.

        Args:
            size_bytes: Size in bytes to release
        """
        self._current_memory = max(0, self._current_memory - size_bytes)

    async def force_cleanup(self) -> None:
        """Force cleanup of all resources."""
        await self.cleanup_tasks()
        self._current_memory = 0
        gc.collect()

    @property
    def active_tasks(self) -> int:
        """Get number of active tasks.

        Returns:
            Number of active tasks
        """
        return len(self._tasks)

    @property
    def current_memory(self) -> int:
        """Get current tracked memory usage in bytes.

        Returns:
            Current memory usage in bytes
        """
        return self._current_memory
