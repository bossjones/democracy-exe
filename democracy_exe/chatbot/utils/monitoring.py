"""Monitoring utilities for tracking system health and resource usage.

This module provides utilities for monitoring tasks, threads, and resources,
including memory usage, thread lifecycle events, and system health metrics.
"""
from __future__ import annotations

import asyncio
import gc
import os
import sys
import threading
import time
import traceback
import weakref

from typing import Any, Dict, Optional, Set

import psutil
import structlog

from democracy_exe.aio_settings import aiosettings


logger = structlog.get_logger(__name__)

class TaskMonitor:
    """Monitor for tracking task lifecycle and resource usage."""

    def __init__(self) -> None:
        """Initialize the task monitor."""
        self._tasks = weakref.WeakSet()
        self._lock = asyncio.Lock()
        self._process = psutil.Process(os.getpid())
        self._start_time = time.monotonic()

    async def register_task(self, task: asyncio.Task) -> None:
        """Register a task for monitoring.

        Args:
            task: The task to monitor
        """
        async with self._lock:
            self._tasks.add(task)
            task.add_done_callback(self._task_done)
            logger.info("Task registered",
                       task_id=id(task),
                       task_name=task.get_name(),
                       active_tasks=len(self._tasks))

    def _task_done(self, task: asyncio.Task) -> None:
        """Handle task completion.

        Args:
            task: The completed task
        """
        try:
            exc = task.exception()
            duration = time.monotonic() - self._start_time
            if exc:
                logger.error("Task failed",
                           task_id=id(task),
                           task_name=task.get_name(),
                           duration=duration,
                           error=str(exc))
            else:
                logger.info("Task completed",
                          task_id=id(task),
                          task_name=task.get_name(),
                          duration=duration)
        except asyncio.CancelledError:
            logger.info("Task cancelled",
                       task_id=id(task),
                       task_name=task.get_name())
        except Exception as e:
            logger.error("Error handling task completion",
                        task_id=id(task),
                        task_name=task.get_name(),
                        error=str(e))

    async def get_metrics(self) -> dict[str, Any]:
        """Get current monitoring metrics.

        Returns:
            A dictionary containing monitoring metrics
        """
        async with self._lock:
            active_tasks = len(self._tasks)
            completed_tasks = sum(1 for t in self._tasks if t.done())
            failed_tasks = sum(1 for t in self._tasks
                             if t.done() and t.exception() is not None)

            metrics = {
                "tasks": {
                    "active": active_tasks,
                    "completed": completed_tasks,
                    "failed": failed_tasks
                }
            }

            try:
                with self._process.oneshot():
                    mem_info = self._process.memory_info()
                    metrics["system"] = {
                        "memory": {
                            "rss": mem_info.rss,
                            "vms": mem_info.vms
                        },
                        "cpu": {
                            "percent": self._process.cpu_percent()
                        },
                        "threads": self._process.num_threads()
                    }
            except Exception as e:
                logger.error("Error getting system metrics",
                           error=str(e))

            return metrics

class ThreadMonitor:
    """Monitor for tracking thread lifecycle and health."""

    def __init__(self) -> None:
        """Initialize the thread monitor."""
        self._threads: set[threading.Thread] = set()
        self._lock = threading.Lock()
        self._process = psutil.Process(os.getpid())

    def register_thread(self, thread: threading.Thread) -> None:
        """Register a thread for monitoring.

        Args:
            thread: The thread to monitor
        """
        with self._lock:
            self._threads.add(thread)
            logger.info("Thread registered",
                       thread_id=thread.ident,
                       thread_name=thread.name,
                       active_threads=len(self._threads))

    def unregister_thread(self, thread: threading.Thread) -> None:
        """Unregister a thread from monitoring.

        Args:
            thread: The thread to stop monitoring
        """
        with self._lock:
            self._threads.discard(thread)
            logger.info("Thread unregistered",
                       thread_id=thread.ident,
                       thread_name=thread.name,
                       active_threads=len(self._threads))

    def log_thread_frames(self) -> None:
        """Log the current stack frames of all monitored threads."""
        frames = sys._current_frames()
        main_thread = threading.main_thread()

        with self._lock:
            for thread in self._threads:
                if thread == main_thread:
                    continue
                if thread.ident:
                    stack = frames.get(thread.ident)
                    if stack:
                        formatted_stack = "".join(traceback.format_stack(stack))
                        logger.info("Thread stack",
                                  thread_id=thread.ident,
                                  thread_name=thread.name,
                                  stack=formatted_stack.strip())

    def get_metrics(self) -> dict[str, Any]:
        """Get current thread monitoring metrics.

        Returns:
            A dictionary containing thread metrics
        """
        with self._lock:
            active_threads = len(self._threads)
            alive_threads = sum(1 for t in self._threads if t.is_alive())
            daemon_threads = sum(1 for t in self._threads if t.daemon)

            metrics = {
                "threads": {
                    "active": active_threads,
                    "alive": alive_threads,
                    "daemon": daemon_threads
                }
            }

            try:
                with self._process.oneshot():
                    ctx_switches = self._process.num_ctx_switches()
                    metrics["system"] = {
                        "context_switches": {
                            "voluntary": ctx_switches.voluntary,
                            "involuntary": ctx_switches.involuntary
                        }
                    }
            except Exception as e:
                logger.error("Error getting thread metrics",
                           error=str(e))

            return metrics

class ResourceMonitor:
    """Monitor for tracking resource usage and cleanup."""

    def __init__(self) -> None:
        """Initialize the resource monitor."""
        self._resources = weakref.WeakSet()
        self._lock = asyncio.Lock()
        self._process = psutil.Process(os.getpid())
        self._start_time = time.monotonic()

    async def register_resource(self, resource: Any) -> None:
        """Register a resource for monitoring.

        Args:
            resource: The resource to monitor
        """
        async with self._lock:
            self._resources.add(resource)
            logger.info("Resource registered",
                       resource_id=id(resource),
                       resource_type=type(resource).__name__,
                       active_resources=len(self._resources))

    async def check_leaks(self) -> None:
        """Check for resource leaks."""
        async with self._lock:
            for resource in self._resources:
                if hasattr(resource, "closed") and not resource.closed:
                    logger.warning("Leaked resource detected",
                                 resource_id=id(resource),
                                 resource_type=type(resource).__name__)

    async def get_metrics(self) -> dict[str, Any]:
        """Get current resource monitoring metrics.

        Returns:
            A dictionary containing resource metrics
        """
        async with self._lock:
            active_resources = len(self._resources)
            metrics = {
                "resources": {
                    "active": active_resources,
                    "types": {}
                },
                "system": {
                    "io": {
                        "read_bytes": 0,
                        "write_bytes": 0,
                        "read_count": 0,
                        "write_count": 0
                    },
                    "open_files": 0,
                    "network_connections": 0
                }
            }

            # Count resources by type
            type_counts: dict[str, int] = {}
            for resource in self._resources:
                resource_type = type(resource).__name__
                type_counts[resource_type] = type_counts.get(resource_type, 0) + 1
            metrics["resources"]["types"] = type_counts

            try:
                with self._process.oneshot():
                    # Get disk I/O statistics
                    try:
                        io_counters = self._process.io_counters()
                        metrics["system"]["io"].update({
                            "read_bytes": io_counters.read_bytes,
                            "write_bytes": io_counters.write_bytes,
                            "read_count": io_counters.read_count,
                            "write_count": io_counters.write_count
                        })
                    except (psutil.Error, AttributeError) as e:
                        logger.warning("Error getting I/O metrics",
                                     error=str(e))

                    # Get open files
                    try:
                        open_files = self._process.open_files()
                        metrics["system"]["open_files"] = len(open_files)
                    except (psutil.Error, AttributeError) as e:
                        logger.warning("Error getting open files metrics",
                                     error=str(e))

                    # Get network connections
                    try:
                        net_connections = self._process.net_connections()
                        metrics["system"]["network_connections"] = len(net_connections)
                    except (psutil.Error, AttributeError) as e:
                        logger.warning("Error getting network connection metrics",
                                     error=str(e))

            except Exception as e:
                logger.error("Error getting system metrics",
                           error=str(e))

            return metrics
