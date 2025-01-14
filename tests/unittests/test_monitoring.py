"""Tests for the monitoring module."""

from __future__ import annotations

import asyncio
import threading
import time

from collections.abc import AsyncGenerator, Generator

import structlog

import pytest

from democracy_exe.chatbot.utils.monitoring import ResourceMonitor, TaskMonitor, ThreadMonitor


@pytest.fixture
async def task_monitor() -> AsyncGenerator[TaskMonitor, None]:
    """Create a task monitor instance for testing.

    Yields:
        TaskMonitor instance
    """
    monitor = TaskMonitor()
    yield monitor


@pytest.fixture
def thread_monitor() -> Generator[ThreadMonitor, None, None]:
    """Create a thread monitor instance for testing.

    Yields:
        ThreadMonitor instance
    """
    monitor = ThreadMonitor()
    yield monitor


@pytest.fixture
async def resource_monitor() -> AsyncGenerator[ResourceMonitor, None]:
    """Create a resource monitor instance for testing.

    Yields:
        ResourceMonitor instance
    """
    monitor = ResourceMonitor()
    yield monitor


async def dummy_task() -> None:
    """Dummy task for testing."""
    await asyncio.sleep(0.1)


async def failing_task() -> None:
    """Task that raises an exception for testing."""
    await asyncio.sleep(0.1)
    raise ValueError("Test error")


@pytest.mark.asyncio
async def test_task_monitor_registration(task_monitor: TaskMonitor) -> None:
    """Test task registration and monitoring.

    Args:
        task_monitor: The task monitor fixture
    """
    task = asyncio.create_task(dummy_task())
    await task_monitor.register_task(task)

    metrics = await task_monitor.get_metrics()
    assert metrics["tasks"]["active"] == 1
    assert metrics["tasks"]["completed"] == 0
    assert metrics["tasks"]["failed"] == 0

    await task
    metrics = await task_monitor.get_metrics()
    assert metrics["tasks"]["completed"] == 1
    assert metrics["tasks"]["failed"] == 0


@pytest.mark.asyncio
async def test_task_monitor_failure(task_monitor: TaskMonitor) -> None:
    """Test monitoring of failed tasks.

    Args:
        task_monitor: The task monitor fixture
    """
    task = asyncio.create_task(failing_task())
    await task_monitor.register_task(task)

    with pytest.raises(ValueError, match="Test error"):
        await task

    metrics = await task_monitor.get_metrics()
    assert metrics["tasks"]["failed"] == 1


def test_thread_monitor_registration(thread_monitor: ThreadMonitor) -> None:
    """Test thread registration and monitoring.

    Args:
        thread_monitor: The thread monitor fixture
    """

    def worker() -> None:
        time.sleep(0.1)

    thread = threading.Thread(target=worker, name="test_thread")
    thread_monitor.register_thread(thread)

    metrics = thread_monitor.get_metrics()
    assert metrics["threads"]["active"] == 1
    assert metrics["threads"]["alive"] == 0  # Not started yet
    assert metrics["threads"]["daemon"] == 0

    thread.start()
    metrics = thread_monitor.get_metrics()
    assert metrics["threads"]["alive"] == 1

    thread.join()
    thread_monitor.unregister_thread(thread)
    metrics = thread_monitor.get_metrics()
    assert metrics["threads"]["active"] == 0


def test_thread_monitor_stack_trace(thread_monitor: ThreadMonitor) -> None:
    """Test thread stack trace logging.

    Args:
        thread_monitor: The thread monitor fixture
    """
    event = threading.Event()

    def worker() -> None:
        while not event.is_set():
            time.sleep(0.1)

    thread = threading.Thread(target=worker, name="test_thread")
    thread_monitor.register_thread(thread)
    thread.start()

    # Give the thread time to start
    time.sleep(0.2)
    thread_monitor.log_thread_frames()

    # Cleanup
    event.set()
    thread.join()
    thread_monitor.unregister_thread(thread)


@pytest.mark.asyncio
async def test_resource_monitor_registration(resource_monitor: ResourceMonitor) -> None:
    """Test resource registration and monitoring.

    Args:
        resource_monitor: The resource monitor fixture
    """

    class DummyResource:
        def __init__(self) -> None:
            self.closed = False

    resource = DummyResource()
    await resource_monitor.register_resource(resource)

    metrics = await resource_monitor.get_metrics()
    assert metrics["resources"]["active"] == 1
    assert metrics["resources"]["types"]["DummyResource"] == 1


@pytest.mark.asyncio
async def test_resource_monitor_leak_detection(resource_monitor: ResourceMonitor) -> None:
    """Test resource leak detection.

    Args:
        resource_monitor: The resource monitor fixture
    """

    class LeakyResource:
        def __init__(self) -> None:
            self.closed = False

    resource = LeakyResource()
    await resource_monitor.register_resource(resource)
    await resource_monitor.check_leaks()  # Should log a warning

    metrics = await resource_monitor.get_metrics()
    assert "io" in metrics["system"]
    assert "open_files" in metrics["system"]
    assert "network_connections" in metrics["system"]
