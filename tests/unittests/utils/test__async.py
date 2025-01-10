"""Tests for async utilities with comprehensive coverage."""

from __future__ import annotations

import asyncio
import os
import threading
import time
import typing

from collections.abc import AsyncGenerator, Generator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Any

import pytest_asyncio
import structlog

import pytest

from pytest_mock import MockerFixture

from democracy_exe import aio_settings
from democracy_exe.utils import async_


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

IS_RUNNING_ON_GITHUB_ACTIONS = bool(os.environ.get("GITHUB_ACTOR"))


@pytest.fixture(autouse=True)
def configure_structlog(log_output: Any) -> None:
    """Configure structlog for testing.

    Args:
        log_output: Log output fixture
    """
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            log_output,
        ],
        wrapper_class=structlog.make_filtering_bound_logger("DEBUG"),
        context_class=dict,
        logger_factory=structlog.testing.LogCapture,
        cache_logger_on_first_use=True,
    )


@pytest.mark.asyncio(scope="module")
@pytest.mark.unittest()
@pytest.mark.integration()
class TestUtilsAsync:
    """Test async utilities with comprehensive coverage."""

    loop: asyncio.AbstractEventLoop

    async def test_remember_loop(self) -> None:
        """Test loop storage for shared tests."""
        TestUtilsAsync.loop = asyncio.get_running_loop()
        assert isinstance(TestUtilsAsync.loop, asyncio.AbstractEventLoop)

    async def test_thread_safe_event(self) -> None:
        """Test ThreadSafeEvent functionality."""
        event = async_.ThreadSafeEvent()

        # Test initial state
        assert not event._is_set
        assert not event._closed

        # Test set/wait
        threading.Thread(target=lambda: time.sleep(0.1) or event.set()).start()
        await event.wait()
        assert event._is_set

        # Test clear
        event.clear()
        assert not event._is_set

        # Test close
        event.close()
        assert event._closed
        with pytest.raises(RuntimeError, match="Event is closed"):
            await event.wait()

    async def test_async_semaphore(self) -> None:
        """Test AsyncSemaphore functionality."""
        sem = async_.AsyncSemaphore(2)

        # Test initial state
        assert sem._count == 2
        assert not sem._closed

        # Test acquire/release
        assert await sem.acquire()
        assert sem._count == 1
        sem.release()
        assert sem._count == 2

        # Test context manager
        async with sem:
            assert sem._count == 1
        assert sem._count == 2

        # Test close
        sem.close()
        with pytest.raises(RuntimeError, match="Semaphore is closed"):
            await sem.acquire()

    async def test_thread_pool_manager(self) -> None:
        """Test ThreadPoolManager functionality."""
        manager = async_.ThreadPoolManager()

        # Test pool creation
        pool1 = manager.create_pool(max_workers=2)
        pool2 = manager.create_pool(max_workers=2)
        assert isinstance(pool1, ThreadPoolExecutor)
        assert isinstance(pool2, ThreadPoolExecutor)

        # Test shutdown
        manager.shutdown_all(wait=True)
        assert manager._closed
        with pytest.raises(RuntimeError, match="ThreadPoolManager is closed"):
            manager.create_pool()

    async def test_gather_with_concurrency(self) -> None:
        """Test gather_with_concurrency functionality."""

        async def slow_task(i: int) -> int:
            await asyncio.sleep(0.1)
            return i

        tasks = [slow_task(i) for i in range(5)]

        # Test successful gathering
        results = await async_.gather_with_concurrency(2, *tasks)
        assert results == list(range(5))

        # Test with failing task
        async def failing_task() -> None:
            raise ValueError("Task failed")

        with pytest.raises(ValueError, match="Task failed"):
            await async_.gather_with_concurrency(2, failing_task())

        # Test with return_exceptions
        results = await async_.gather_with_concurrency(2, failing_task(), return_exceptions=True)
        assert isinstance(results[0], ValueError)

    async def test_to_async(self) -> None:
        """Test to_async decorator."""

        @async_.to_async
        def sync_func(x: int) -> int:
            time.sleep(0.1)
            return x * 2

        # Test successful execution
        result = await sync_func(21)
        assert result == 42

        # Test error handling
        @async_.to_async
        def failing_func() -> None:
            raise ValueError("Sync error")

        with pytest.raises(ValueError, match="Sync error"):
            await failing_func()

    async def test_to_async_thread(self) -> None:
        """Test to_async_thread decorator."""

        @async_.to_async_thread
        def thread_func(x: int) -> int:
            time.sleep(0.1)
            return x * 2

        # Test successful execution
        result = await thread_func(21)
        assert result == 42

        # Test error handling
        @async_.to_async_thread
        def failing_thread_func() -> None:
            raise ValueError("Thread error")

        with pytest.raises(ValueError, match="Thread error"):
            await failing_thread_func()

    def test_to_sync(self) -> None:
        """Test to_sync decorator."""

        @async_.to_sync
        async def async_func(x: int) -> int:
            await asyncio.sleep(0.1)
            return x * 2

        # Test successful execution
        result = async_func(21)
        assert result == 42

        # Test error handling
        @async_.to_sync
        async def failing_async_func() -> None:
            raise ValueError("Async error")

        with pytest.raises(ValueError, match="Async error"):
            failing_async_func()

    async def test_fire_coroutine_threadsafe(self) -> None:
        """Test fire_coroutine_threadsafe functionality."""
        loop = asyncio.get_running_loop()
        event = asyncio.Event()

        async def coro() -> None:
            event.set()

        # Test successful execution
        async_.fire_coroutine_threadsafe(coro(), loop)
        await asyncio.wait_for(event.wait(), timeout=1)
        assert event.is_set()

        # Test error cases
        with pytest.raises(TypeError, match="A coroutine object is required"):
            async_.fire_coroutine_threadsafe(lambda: None, loop)  # type: ignore

    async def test_run_callback_threadsafe(self) -> None:
        """Test run_callback_threadsafe functionality."""
        loop = asyncio.get_running_loop()

        def callback(x: int) -> int:
            return x * 2

        # Test successful execution
        future = async_.run_callback_threadsafe(loop, callback, 21)
        result = await asyncio.wrap_future(future)
        assert result == 42

        # Test error handling
        def failing_callback() -> None:
            raise ValueError("Callback error")

        future = async_.run_callback_threadsafe(loop, failing_callback)
        with pytest.raises(ValueError, match="Callback error"):
            await asyncio.wrap_future(future)

    def test_check_loop(self) -> None:
        """Test check_loop functionality."""
        # Test outside event loop
        async_.check_loop()  # Should not raise

        # Test inside event loop
        async def inside_loop() -> None:
            async_.check_loop()

        with pytest.raises(RuntimeError, match="Detected I/O inside the event loop"):
            asyncio.run(inside_loop())

    def test_protect_loop(self) -> None:
        """Test protect_loop decorator."""

        @async_.protect_loop
        def protected_func() -> str:
            return "success"

        # Test outside event loop
        assert protected_func() == "success"

        # Test inside event loop
        @async_.protect_loop
        def inside_loop_func() -> None:
            pass

        async def run_protected() -> None:
            inside_loop_func()

        with pytest.raises(RuntimeError, match="Detected I/O inside the event loop"):
            asyncio.run(run_protected())

    async def test_async_timed(self) -> None:
        """Test async_timed decorator."""

        @async_.async_timed()
        async def timed_func() -> str:
            await asyncio.sleep(0.1)
            return "done"

        with structlog.testing.capture_logs() as logs:
            result = await timed_func()
            assert result == "done"
            assert any("Starting timed_func" in log.get("event", "") for log in logs)
            assert any("Finished timed_func" in log.get("event", "") for log in logs)

    async def test_async_timer(self) -> None:
        """Test async_timer decorator."""

        @async_.async_timer()
        async def timed_func() -> str:
            await asyncio.sleep(0.1)
            return "done"

        with structlog.testing.capture_logs() as logs:
            result = await timed_func()
            assert result == "done"
            assert any("Starting timed_func" in log.get("event", "") for log in logs)
            assert any("Finished timed_func" in log.get("event", "") for log in logs)

    async def test_shutdown_run_callback_threadsafe(self) -> None:
        """Test shutdown_run_callback_threadsafe functionality."""
        loop = asyncio.get_running_loop()

        # Test normal operation
        future = async_.run_callback_threadsafe(loop, lambda: 42)
        result = await asyncio.wrap_future(future)
        assert result == 42

        # Test shutdown
        async_.shutdown_run_callback_threadsafe(loop)
        with pytest.raises(RuntimeError, match="event loop is in the process of shutting down"):
            async_.run_callback_threadsafe(loop, lambda: None)

    @pytest.mark.asyncio
    async def test_cleanup_thread_pools(self) -> None:
        """Test cleanup_thread_pools functionality."""
        # Create some thread pools
        pool1 = ThreadPoolExecutor(max_workers=1)
        pool2 = ThreadPoolExecutor(max_workers=1)

        async_.register_thread_pool(pool1)
        async_.register_thread_pool(pool2)

        # Test cleanup
        async_.cleanup_thread_pools()

        # Verify pools are shut down
        assert pool1._shutdown
        assert pool2._shutdown

    async def test_thread_safe_event_concurrent(self) -> None:
        """Test ThreadSafeEvent under concurrent access."""
        event = async_.ThreadSafeEvent()
        results: list[bool] = []

        async def waiter() -> None:
            try:
                await asyncio.wait_for(event.wait(), timeout=1.0)
                results.append(True)
            except TimeoutError:
                results.append(False)

        # Start multiple waiters
        waiters = [asyncio.create_task(waiter()) for _ in range(5)]

        # Set event after small delay
        await asyncio.sleep(0.1)
        event.set()

        # Wait for all waiters
        await asyncio.gather(*waiters)
        assert all(results)  # All waiters should succeed

    async def test_async_semaphore_concurrent(self) -> None:
        """Test AsyncSemaphore under concurrent access."""
        sem = async_.AsyncSemaphore(2)
        counter = 0
        max_concurrent = 0

        async def worker() -> None:
            nonlocal counter, max_concurrent
            async with sem:
                counter += 1
                max_concurrent = max(max_concurrent, counter)
                await asyncio.sleep(0.1)
                counter -= 1

        # Run multiple workers
        workers = [worker() for _ in range(5)]
        await asyncio.gather(*workers)

        assert max_concurrent <= 2  # Never exceeded semaphore limit

    async def test_thread_pool_manager_exception(self) -> None:
        """Test ThreadPoolManager error handling and cleanup."""
        manager = async_.ThreadPoolManager()
        pool = manager.create_pool(max_workers=1)

        def failing_task() -> None:
            raise ValueError("Task failed")

        # Submit failing task
        future = pool.submit(failing_task)
        with pytest.raises(ValueError, match="Task failed"):
            future.result()

        # Verify pool is still usable
        assert not pool._shutdown

        # Test cleanup during exception
        try:
            raise RuntimeError("Forced error")
        except RuntimeError:
            manager.shutdown_all(wait=True)
            assert manager._closed

    async def test_gather_with_concurrency_cancel(self) -> None:
        """Test gather_with_concurrency cancellation handling."""

        async def slow_task(i: int) -> int:
            try:
                await asyncio.sleep(10)
                return i
            except asyncio.CancelledError:
                raise

        tasks = [slow_task(i) for i in range(3)]
        gather_task = asyncio.create_task(async_.gather_with_concurrency(2, *tasks))

        # Cancel after small delay
        await asyncio.sleep(0.1)
        gather_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await gather_task

    async def test_timeout_handling(self) -> None:
        """Test timeout handling in async operations."""
        event = async_.ThreadSafeEvent()

        # Test wait with timeout
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(event.wait(), timeout=0.1)

        # Test semaphore timeout
        sem = async_.AsyncSemaphore(1)
        async with sem:  # Hold the semaphore
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(sem.acquire(), timeout=0.1)

    async def test_cleanup_during_exception(self) -> None:
        """Test resource cleanup during exceptions."""
        event = async_.ThreadSafeEvent()
        sem = async_.AsyncSemaphore(1)
        manager = async_.ThreadPoolManager()

        try:
            raise RuntimeError("Forced cleanup")
        except RuntimeError:
            # Verify resources can be cleaned up
            event.close()
            sem.close()
            manager.shutdown_all(wait=True)

            assert event._closed
            assert sem._closed
            assert manager._closed

    async def test_thread_pool_manager_max_workers(self) -> None:
        """Test ThreadPoolManager with different worker configurations."""
        manager = async_.ThreadPoolManager()

        # Test with default workers
        pool1 = manager.create_pool()
        assert pool1._max_workers == (os.cpu_count() or 1)

        # Test with custom workers
        pool2 = manager.create_pool(max_workers=2)
        assert pool2._max_workers == 2

        # Test with too many workers
        pool3 = manager.create_pool(max_workers=1000)
        assert pool3._max_workers <= (os.cpu_count() or 1) * 5  # Should be capped

        manager.shutdown_all(wait=True)
