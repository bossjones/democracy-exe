"""Tests for the resource manager module."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import resource
import sys
import weakref

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

import psutil
import pytest_structlog
import structlog

import pytest

from democracy_exe.chatbot.utils.resource_manager import ResourceLimits, ResourceManager


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def resource_limits() -> ResourceLimits:
    """Create test resource limits.

    Returns:
        ResourceLimits: Test resource limits configuration
    """
    return ResourceLimits(
        max_memory_mb=128, max_tasks=5, max_response_size_mb=1, max_buffer_size_kb=32, task_timeout_seconds=1
    )


@pytest.fixture
def resource_manager(resource_limits: ResourceLimits) -> ResourceManager:
    """Create test resource manager.

    Args:
        resource_limits: Test resource limits

    Returns:
        ResourceManager: Test resource manager instance
    """
    return ResourceManager(limits=resource_limits)


@pytest.fixture(autouse=True)
def configure_structlog() -> None:
    """Configure structlog for testing.

    This fixture configures structlog with the necessary processors
    for testing, including the pytest_structlog capture processor.
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.testing.LogCapture(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


@pytest.mark.asyncio
async def test_check_memory(
    resource_manager: ResourceManager, mocker: MockerFixture, log: pytest_structlog.StructuredLogCapture
) -> None:
    """Test memory usage checking.

    Args:
        resource_manager: Test resource manager
        mocker: Pytest mocker
        log: Structlog capture fixture
    """
    # Mock memory info
    mock_memory = mocker.MagicMock()
    mock_memory.rss = 64 * 1024 * 1024  # 64MB
    mock_memory.vms = 128 * 1024 * 1024  # 128MB VMS
    mocker.patch.object(resource_manager._process, "memory_info", return_value=mock_memory)
    mocker.patch.object(resource_manager._process, "num_threads", return_value=10)
    mocker.patch.object(resource_manager._process, "cpu_percent", return_value=0.0)

    # Test memory within limits
    with structlog.testing.capture_logs() as captured:
        assert await resource_manager.check_memory() is True

        # Verify debug logs
        assert any(
            log.get("event") == "Checking memory usage"
            and log.get("log_level") == "debug"
            and log.get("memory_bytes") == 64 * 1024 * 1024
            and log.get("cpu_percent") == 0.0
            for log in captured
        ), "Expected memory check debug log not found"

        assert any(
            log.get("event") == "Memory check passed"
            and log.get("log_level") == "debug"
            and log.get("headroom_mb") == "64.0"
            for log in captured
        ), "Expected memory check passed log not found"

    # Test memory exceeds limits
    mock_memory.rss = 256 * 1024 * 1024  # 256MB
    with structlog.testing.capture_logs() as captured:
        assert await resource_manager.check_memory() is False

        # Verify error log
        assert any(
            log.get("event") == "Memory usage exceeds limit" and log.get("log_level") == "error" for log in captured
        ), "Expected memory limit exceeded log not found"


@pytest.mark.asyncio
async def test_track_task(resource_manager: ResourceManager) -> None:
    """Test task tracking.

    Args:
        resource_manager: Test resource manager
    """
    # Create test tasks
    tasks = []
    for i in range(5):
        task = asyncio.create_task(asyncio.sleep(0.1))
        await resource_manager.track_task(task)
        tasks.append(task)

    assert resource_manager.active_tasks == 5

    # Test task limit exceeded
    with pytest.raises(RuntimeError, match="Max concurrent tasks limit"):
        task = asyncio.create_task(asyncio.sleep(0.1))
        await resource_manager.track_task(task)

    # Clean up tasks
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_cleanup_tasks(resource_manager: ResourceManager, mocker: MockerFixture) -> None:
    """Test task cleanup.

    Args:
        resource_manager: Test resource manager
        mocker: Pytest mocker
    """
    # Create test tasks
    tasks = []
    for i in range(3):
        task = asyncio.create_task(asyncio.sleep(2))
        await resource_manager.track_task(task)
        tasks.append(task)

    # Test cleanup
    await resource_manager.cleanup_tasks()
    assert all(task.cancelled() for task in tasks)
    assert resource_manager.active_tasks == 0


@pytest.mark.asyncio
async def test_memory_tracking(resource_manager: ResourceManager, log: pytest_structlog.StructuredLogCapture) -> None:
    """Test memory allocation tracking.

    Args:
        resource_manager: Test resource manager
        log: Structlog capture fixture
    """
    # Track memory allocation
    size = 32 * 1024 * 1024  # 32MB

    with structlog.testing.capture_logs() as captured:
        resource_manager.track_memory(size)

        # Verify tracking log
        assert any(
            log.get("event") == "Memory tracked"
            and log.get("log_level") == "debug"
            and log.get("allocation") == size
            and log.get("total") == size
            for log in captured
        ), "Expected memory tracking log not found"

    # Test memory limit exceeded
    large_size = 256 * 1024 * 1024  # 256MB
    with pytest.raises(RuntimeError, match="Memory allocation.*would exceed limit"):
        with structlog.testing.capture_logs() as captured:
            resource_manager.track_memory(large_size)

            # Verify error log
            assert any(
                log.get("event") == "Memory allocation would exceed limit"
                and log.get("log_level") == "error"
                and log.get("allocation") == large_size
                for log in captured
            ), "Expected memory limit error log not found"

    # Release memory
    with structlog.testing.capture_logs() as captured:
        resource_manager.release_memory(size)

        # Verify release log
        assert any(
            log.get("event") == "Memory released"
            and log.get("log_level") == "debug"
            and log.get("release") == size
            and log.get("remaining") == 0
            for log in captured
        ), "Expected memory release log not found"

    assert resource_manager.current_memory_usage == 0


@pytest.mark.asyncio
async def test_force_cleanup(
    resource_manager: ResourceManager, mocker: MockerFixture, log: pytest_structlog.StructuredLogCapture
) -> None:
    """Test force cleanup of resources.

    Args:
        resource_manager: Test resource manager
        mocker: Pytest mocker
        log: Structlog capture fixture
    """
    # Mock cleanup methods
    mock_gc = mocker.patch("gc.collect")
    mock_setrlimit = mocker.patch("resource.setrlimit")

    # Create test task and allocate memory
    task = asyncio.create_task(asyncio.sleep(1))
    await resource_manager.track_task(task)
    resource_manager.track_memory(32 * 1024 * 1024)  # Track 32MB

    # Test cleanup
    await resource_manager.force_cleanup()

    # Verify cleanup logs
    assert log.has("Task cleanup timed out or cancelled", level="warning"), "Expected task cleanup log not found"
    assert log.has("Tasks cleaned up", level="debug"), "Expected tasks cleaned up log not found"
    assert log.has("Forced cleanup completed", level="info", remaining_tasks=0, memory_usage=0), (
        "Expected force cleanup completion log not found"
    )

    # Verify cleanup effects
    assert task.cancelled()
    assert resource_manager.active_tasks == 0
    assert resource_manager.current_memory_usage == 0
    mock_gc.assert_called_once()

    if sys.platform != "win32":
        mock_setrlimit.assert_called_once_with(
            resource.RLIMIT_AS,
            (resource_manager.limits.max_memory_mb * 1024 * 1024, resource_manager.limits.max_memory_mb * 1024 * 1024),
        )
