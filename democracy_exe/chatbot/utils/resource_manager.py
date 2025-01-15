"""Resource management utilities.

This module provides utilities for managing system resources,
including memory usage and task tracking.
"""
from __future__ import annotations

import asyncio
import gc
import os

from dataclasses import dataclass
from typing import Optional, Set

import psutil
import structlog


logger = structlog.get_logger(__name__)

@dataclass
class ResourceLimits:
    """Resource limits configuration.

    Attributes:
        max_memory_mb: Maximum memory usage in MB
        max_tasks: Maximum number of concurrent tasks
        max_response_size_mb: Maximum response size in MB
        max_buffer_size_kb: Maximum buffer size in KB
        task_timeout_seconds: Task timeout in seconds
    """
    max_memory_mb: int
    max_tasks: int
    max_response_size_mb: int
    max_buffer_size_kb: int
    task_timeout_seconds: int

class ResourceManager:
    """Manager for system resources.

    This class manages system resources including memory usage and task tracking.
    It provides methods for checking memory usage, tracking tasks, and cleaning up
    resources.

    Attributes:
        limits: Resource limits configuration
        _tasks: Set of active tasks
        _memory_usage: Current memory usage in bytes
    """
    def __init__(self, limits: ResourceLimits) -> None:
        """Initialize the resource manager.

        Args:
            limits: Resource limits configuration
        """
        self._limits = limits
        self._tasks: set[asyncio.Task] = set()
        self._memory_usage = 0
        self._process = psutil.Process(os.getpid())

    @property
    def limits(self) -> ResourceLimits:
        """Get the resource limits configuration.

        Returns:
            ResourceLimits: The resource limits configuration
        """
        return self._limits

    async def check_memory(self) -> None:
        """Check if memory usage is within limits.

        This method checks the current memory usage and raises a RuntimeError
        if it exceeds the configured limit.

        Raises:
            RuntimeError: If memory usage exceeds the limit
        """
        memory_info = self._process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024

        if memory_mb > self._limits.max_memory_mb:
            logger.error("Memory usage exceeds limit",
                        memory_mb=memory_mb,
                        limit=self._limits.max_memory_mb)
            raise RuntimeError(f"Memory usage {memory_mb:.1f}MB exceeds limit {self._limits.max_memory_mb}MB")

    def track_task(self, task: asyncio.Task) -> None:
        """Track a new task.

        This method adds a task to the set of tracked tasks and checks if the
        number of active tasks exceeds the configured limit.

        Args:
            task: The task to track

        Raises:
            RuntimeError: If max concurrent tasks limit is reached
        """
        # Clean up completed tasks
        self._tasks = {t for t in self._tasks if not t.done()}

        # Check task limit
        if len(self._tasks) >= self._limits.max_tasks:
            logger.error("Max concurrent tasks limit reached",
                        active_tasks=len(self._tasks),
                        limit=self._limits.max_tasks)
            raise RuntimeError(f"Max concurrent tasks limit {self._limits.max_tasks} reached")

        self._tasks.add(task)
        logger.debug("Task added to tracking",
                    task_id=id(task),
                    active_tasks=len(self._tasks))

    async def cleanup_tasks(self, tasks: set[asyncio.Task] | None = None) -> None:
        """Clean up tasks.

        This method cancels and cleans up the specified tasks, or all tracked
        tasks if none are specified.

        Args:
            tasks: Optional set of tasks to clean up. If None, all tasks are cleaned up.
        """
        tasks_to_cleanup = tasks or self._tasks.copy()

        for task in tasks_to_cleanup:
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (TimeoutError, asyncio.CancelledError):
                    logger.warning("Task cleanup timed out or cancelled",
                                 task_id=id(task))
                except Exception as e:
                    logger.error("Task cleanup failed",
                               task_id=id(task),
                               error=str(e))

            self._tasks.discard(task)

        logger.debug("Tasks cleaned up",
                    num_tasks=len(tasks_to_cleanup),
                    remaining_tasks=len(self._tasks))

    def track_memory(self, size: int) -> None:
        """Track memory allocation.

        This method tracks memory allocation and checks if it would exceed
        the configured limit.

        Args:
            size: Size of memory to track in bytes

        Raises:
            RuntimeError: If memory allocation would exceed the limit
        """
        new_usage = self._memory_usage + size
        if new_usage > self._limits.max_memory_mb * 1024 * 1024:
            logger.error("Memory allocation would exceed limit",
                        allocation=size,
                        current=self._memory_usage,
                        limit=self._limits.max_memory_mb * 1024 * 1024)
            raise RuntimeError(f"Memory allocation of {size} bytes would exceed limit")

        self._memory_usage = new_usage
        logger.debug("Memory tracked",
                    allocation=size,
                    total=self._memory_usage)

    def release_memory(self, size: int) -> None:
        """Release tracked memory.

        This method releases tracked memory allocation.

        Args:
            size: Size of memory to release in bytes
        """
        self._memory_usage = max(0, self._memory_usage - size)
        logger.debug("Memory released",
                    release=size,
                    remaining=self._memory_usage)

    async def force_cleanup(self) -> None:
        """Force cleanup of all resources.

        This method performs a forced cleanup of all resources, including
        tasks and memory tracking.
        """
        # Clean up all tasks
        await self.cleanup_tasks()

        # Reset memory tracking
        self._memory_usage = 0

        # Force garbage collection
        gc.collect()

        logger.info("Forced cleanup completed",
                   remaining_tasks=len(self._tasks),
                   memory_usage=self._memory_usage)

    @property
    def active_tasks(self) -> int:
        """Get the number of active tasks.

        Returns:
            int: The number of active tasks
        """
        return len(self._tasks)

    @property
    def current_memory_usage(self) -> int:
        """Get the current tracked memory usage.

        Returns:
            int: The current memory usage in bytes
        """
        return self._memory_usage
