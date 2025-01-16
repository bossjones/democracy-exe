"""Tests for the resource manager module."""

from __future__ import annotations

import asyncio
import gc
import os
import resource
import sys
import weakref

from collections.abc import AsyncGenerator, Generator
from typing import TYPE_CHECKING

import psutil
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


@pytest.mark.asyncio
async def test_check_memory(resource_manager: ResourceManager, mocker: MockerFixture) -> None:
    """Test memory usage checking.

    Args:
        resource_manager: Test resource manager
        mocker: Pytest mocker
    """
    # Mock memory info
    mock_memory = mocker.MagicMock()
    mock_memory.rss = 64 * 1024 * 1024  # 64MB
    mocker.patch.object(resource_manager._process, "memory_info", return_value=mock_memory)

    # Test memory within limits
    assert await resource_manager.check_memory()

    # Test memory exceeds limits
    mock_memory.rss = 256 * 1024 * 1024  # 256MB
    assert not await resource_manager.check_memory()


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
async def test_memory_tracking(resource_manager: ResourceManager) -> None:
    """Test memory allocation tracking.

    Args:
        resource_manager: Test resource manager
    """
    # Track memory allocation
    size = 32 * 1024 * 1024  # 32MB
    await resource_manager.track_memory(size)

    # Test memory limit exceeded
    with pytest.raises(RuntimeError, match="Memory limit.*would be exceeded"):
        await resource_manager.track_memory(256 * 1024 * 1024)  # 256MB

    # Release memory
    await resource_manager.release_memory(size)
    assert resource_manager._current_memory_usage == 0


@pytest.mark.asyncio
async def test_force_cleanup(resource_manager: ResourceManager, mocker: MockerFixture) -> None:
    """Test force cleanup of resources.

    Args:
        resource_manager: Test resource manager
        mocker: Pytest mocker
    """
    # Mock cleanup methods
    mock_gc = mocker.patch("gc.collect")
    mock_setrlimit = mocker.patch("resource.setrlimit")

    # Create test task
    task = asyncio.create_task(asyncio.sleep(1))
    await resource_manager.track_task(task)

    # Test cleanup
    await resource_manager.force_cleanup()

    # Verify cleanup
    assert task.cancelled()
    assert resource_manager.active_tasks == 0
    assert resource_manager._current_memory_usage == 0
    mock_gc.assert_called_once()

    if sys.platform != "win32":
        mock_setrlimit.assert_called_once_with(resource.RLIMIT_AS, (128 * 1024 * 1024, 128 * 1024 * 1024))
