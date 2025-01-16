"""Resource management utilities.

This module provides utilities for managing system resources,
including memory usage and task tracking.

<structlog_practices>
    <concurrency>
        - Use contextvars instead of bind() for thread/async-safe context management
        - Clear context at start and end of each method using clear_contextvars()
        - Bind context using bind_contextvars() for thread/async safety
        - Context is automatically isolated between different async tasks
        - Storage mechanics differ between concurrency methods (threads vs async)
    </concurrency>

    <performance>
        - Cache loggers on first use with cache_logger_on_first_use=True
        - Create local logger instances for frequent logging
        - Use native BoundLogger with make_filtering_bound_logger() for level filtering
        - Avoid sending logs through stdlib if possible
        - Consider using faster JSON serializers (orjson, msgspec)
        - Be conscious about asyncio support usage and performance impact
    </performance>

    <context_management>
        - Use merge_contextvars processor in configuration
        - Clear context at start of request/method handlers
        - Use bound_contextvars context manager for temporary bindings
        - Access context storage with get_contextvars/get_merged_contextvars
        - Context is isolated between sync and async code
    </context_management>

    <example_configuration>
        structlog.configure(
            cache_logger_on_first_use=True,
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                structlog.processors.JSONRenderer(serializer=orjson.dumps),
            ],
            logger_factory=structlog.BytesLoggerFactory(),
        )
    </example_configuration>
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

        # Clear any existing context and bind new context vars
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            memory_mb=f"{memory_mb:.1f}",
            limit_mb=self._limits.max_memory_mb,
            process_id=self._process.pid
        )

        # Add detailed system metrics at debug level
        logger.debug(
            "Checking memory usage",
            memory_bytes=memory_info.rss,
            virtual_memory=memory_info.vms,
            num_threads=self._process.num_threads(),
            cpu_percent=self._process.cpu_percent(),
            memory_percent=f"{(memory_mb / self._limits.max_memory_mb * 100):.1f}%"
        )

        if memory_mb > self._limits.max_memory_mb:
            # Context vars are automatically included
            logger.error(
                "Memory usage exceeds limit",
                memory_info=memory_info._asdict(),
                headroom_mb=f"{(self._limits.max_memory_mb - memory_mb):.1f}"
            )
            raise RuntimeError(f"Memory usage {memory_mb:.1f}MB exceeds limit {self._limits.max_memory_mb}MB")

        # Context vars are automatically included
        logger.debug(
            "Memory check passed",
            headroom_mb=f"{(self._limits.max_memory_mb - memory_mb):.1f}"
        )

        # Clean up context vars at the end
        structlog.contextvars.clear_contextvars()

    async def track_task(self, task: asyncio.Task) -> None:
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

        # Set context for all log messages in this method
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            active_tasks=len(self._tasks),
            limit=self._limits.max_tasks,
            task_id=id(task)
        )

        # Check task limit
        if len(self._tasks) >= self._limits.max_tasks:
            logger.error("Max concurrent tasks limit reached")
            raise RuntimeError(f"Max concurrent tasks limit {self._limits.max_tasks} reached")

        self._tasks.add(task)
        logger.debug("Task added to tracking")

        # Clean up context
        structlog.contextvars.clear_contextvars()

    async def cleanup_tasks(self, tasks: set[asyncio.Task] | None = None) -> None:
        """Clean up tasks.

        This method cancels and cleans up the specified tasks, or all tracked
        tasks if none are specified.

        Args:
            tasks: Optional set of tasks to clean up. If None, all tasks are cleaned up.
        """
        tasks_to_cleanup = tasks or self._tasks.copy()

        # Set initial context for the cleanup operation
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            num_tasks=len(tasks_to_cleanup),
            remaining_tasks=len(self._tasks)
        )

        for task in tasks_to_cleanup:
            # Update context for each task
            structlog.contextvars.bind_contextvars(task_id=id(task))

            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (TimeoutError, asyncio.CancelledError):
                    logger.warning("Task cleanup timed out or cancelled")
                except Exception as e:
                    logger.error("Task cleanup failed", error=str(e))

            self._tasks.discard(task)

        logger.debug("Tasks cleaned up")

        # Clean up context at the end
        structlog.contextvars.clear_contextvars()

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
