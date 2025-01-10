# pylint: disable=no-member
"""Tests for async utilities with comprehensive coverage."""

from __future__ import annotations

import asyncio
import logging
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
def configure_structlog() -> None:
    """Configure structlog for testing."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.unittest()
@pytest.mark.integration()
class TestUtilsAsync:
    """Test async utilities with comprehensive coverage."""

    # Class variable to store the loop for all tests
    _loop: asyncio.AbstractEventLoop | None = None

    async def test_remember_loop(self) -> None:
        """Test loop storage for shared tests."""
        current_loop = asyncio.get_running_loop()
        assert isinstance(current_loop, asyncio.AbstractEventLoop)

        # Store the loop for other tests to use
        TestUtilsAsync._loop = current_loop

        # Verify we have a valid running loop
        assert current_loop.is_running()

    async def test_thread_safe_event(self) -> None:
        """Test ThreadSafeEvent functionality with comprehensive coverage.

        This test verifies:
        1. Initial state of the event
        2. Setting event from another thread
        3. Multiple waiters receiving the event
        4. Clearing event from another thread
        5. Error handling after close
        6. Thread safety of all operations
        """
        # Create event in async context to get the event loop
        event = async_.ThreadSafeEvent()

        # Test initial state
        assert not event._is_set
        assert not event._closed
        assert event._loop is not None
        assert isinstance(event._loop, asyncio.AbstractEventLoop)

        # Test set/wait with multiple waiters
        results: list[bool] = []
        waiters = []
        ready_event = threading.Event()  # Synchronize thread start

        async def waiter() -> None:
            """Wait for the event with timeout."""
            try:
                await asyncio.wait_for(event.wait(), timeout=2.0)
                results.append(True)
            except TimeoutError:
                results.append(False)

        # Start multiple waiters
        for _ in range(5):
            waiters.append(asyncio.create_task(waiter()))

        # Test set from another thread
        def thread_set() -> None:
            """Set the event from a background thread."""
            ready_event.wait(timeout=1.0)  # Wait for main thread signal
            event.set()

        thread = threading.Thread(target=thread_set)
        thread.start()

        # Signal thread to proceed and wait for waiters
        ready_event.set()
        await asyncio.gather(*waiters)
        thread.join()

        # Verify all waiters succeeded
        assert all(results), "All waiters should have received the event"
        assert event._is_set, "Event should still be set"

        # Test clear from another thread
        clear_done = threading.Event()

        def thread_clear() -> None:
            """Clear the event from a background thread."""
            event.clear()
            clear_done.set()

        thread = threading.Thread(target=thread_clear)
        thread.start()
        clear_done.wait(timeout=1.0)  # Wait for clear to complete
        thread.join()

        assert not event._is_set, "Event should be cleared"

        # Test error handling in set/clear after close
        event.close()
        assert event._closed, "Event should be closed"
        assert event._event is None, "Internal event should be None"
        assert event._loop is None, "Event loop reference should be cleared"

        # Set/clear should not raise after close
        event.set()  # Should return silently
        event.clear()  # Should return silently

        # Wait should raise after close
        with pytest.raises(RuntimeError, match="Event is closed"):
            await event.wait()

        # Test creating event in non-async context
        def create_event_in_thread() -> None:
            """Try to create event in non-async context."""
            with pytest.raises(RuntimeError, match="must be created from an async context"):
                async_.ThreadSafeEvent()

        thread = threading.Thread(target=create_event_in_thread)
        thread.start()
        thread.join()

    async def test_async_semaphore(self) -> None:
        """Test AsyncSemaphore functionality."""
        # Initialize semaphore with value 2
        sem = async_.AsyncSemaphore(2)
        assert isinstance(sem._lock, type(threading.Lock()))
        assert sem._count == 2

        # Test basic acquire/release
        acquired = await sem.acquire()
        assert acquired
        assert sem._count == 1
        sem.release()
        assert sem._count == 2

        # Test multiple concurrent waiters
        waiters = []
        results: list[bool] = []
        ready_event = threading.Event()

        async def waiter() -> None:
            """Wait for semaphore with timeout."""
            try:
                await asyncio.wait_for(sem.acquire(), timeout=2.0)
                results.append(True)
                await asyncio.sleep(0.1)  # Hold the semaphore for a bit
                sem.release()
            except TimeoutError:
                results.append(False)

        # Start multiple waiters (more than semaphore value)
        for _ in range(5):
            waiters.append(asyncio.create_task(waiter()))

        # Let waiters run
        await asyncio.sleep(0.1)

        # Release semaphore from another thread
        def thread_release() -> None:
            """Release semaphore from background thread."""
            ready_event.wait(timeout=1.0)
            for _ in range(3):  # Release for remaining waiters
                try:
                    sem.release()
                except RuntimeError:
                    pass  # Ignore if already released

        thread = threading.Thread(target=thread_release)
        thread.start()

        # Signal thread to proceed and wait for waiters
        ready_event.set()
        await asyncio.gather(*waiters)
        thread.join()

        # Verify all waiters succeeded
        assert all(results), "All waiters should have acquired the semaphore"
        assert sem._count == 2, "Semaphore should be fully released"

        # Test closing semaphore
        sem.close()
        with pytest.raises(RuntimeError, match="Semaphore is closed"):
            await sem.acquire()

        # Test error on release after close
        with pytest.raises(RuntimeError, match="Semaphore is closed"):
            sem.release()

    async def test_thread_pool_manager(self) -> None:
        """Test ThreadPoolManager functionality with comprehensive coverage."""
        manager = async_.ThreadPoolManager()

        # Test pool creation with different worker configs
        pool1 = manager.create_pool(max_workers=2)
        pool2 = manager.create_pool(max_workers=3)
        assert isinstance(pool1, ThreadPoolExecutor)
        assert isinstance(pool2, ThreadPoolExecutor)
        assert pool1._max_workers == 2
        assert pool2._max_workers == 3

        # Test worker task execution
        def worker_task() -> str:
            return "success"

        future = pool1.submit(worker_task)
        assert future.result() == "success"

        # Test error propagation
        def failing_task() -> None:
            raise ValueError("Task failed")

        future = pool1.submit(failing_task)
        with pytest.raises(ValueError, match="Task failed"):
            future.result()

        # Verify pool is still usable after error
        assert not pool1._shutdown

        # Test proper cleanup during shutdown
        manager.shutdown_all(wait=True)
        assert manager._closed
        assert pool1._shutdown
        assert pool2._shutdown

        # Verify cannot create new pools after shutdown
        with pytest.raises(RuntimeError, match="ThreadPoolManager is closed"):
            manager.create_pool()

        # Test cleanup during exception
        manager = async_.ThreadPoolManager()
        pool = manager.create_pool(max_workers=1)
        try:
            raise RuntimeError("Forced error")
        except RuntimeError:
            manager.shutdown_all(wait=True)
            assert manager._closed
            assert pool._shutdown

    async def test_gather_with_concurrency(self) -> None:
        """Test gather_with_concurrency functionality with comprehensive coverage.

        Tests:
        1. Basic concurrent task execution
        2. Concurrency limit enforcement
        3. Error propagation
        4. Task cancellation
        5. Invalid concurrency values
        6. Mixed success/failure scenarios
        7. Resource cleanup
        """
        # Track task execution times for concurrency verification
        execution_times: list[float] = []

        async def tracked_task(i: int, delay: float = 0.1) -> int:
            """Task that tracks its execution time."""
            start = time.time()
            try:
                await asyncio.sleep(delay)
                execution_times.append(time.time() - start)
                return i
            except asyncio.CancelledError:
                execution_times.append(time.time() - start)
                raise

        # Test invalid concurrency limit
        task = tracked_task(1)
        with pytest.raises(ValueError, match="Concurrency limit must be >= 1"):
            await async_.gather_with_concurrency(0, task)
        # Clean up unused coroutine
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Test basic concurrent execution
        tasks = [tracked_task(i) for i in range(5)]
        results = await async_.gather_with_concurrency(2, *tasks)
        assert results == list(range(5))

        # Verify concurrency limit was respected
        # With 5 tasks, 0.1s delay, and concurrency 2, should take ~0.3s
        assert len([t for t in execution_times if t >= 0.1]) >= 3
        execution_times.clear()

        # Test mixed success/failure scenario
        async def failing_task() -> None:
            await asyncio.sleep(0.1)
            raise ValueError("Task failed")

        mixed_tasks = [tracked_task(1), failing_task(), tracked_task(2)]
        with pytest.raises(ValueError, match="Task failed"):
            await async_.gather_with_concurrency(2, *mixed_tasks)

        # Test with return_exceptions
        results = await async_.gather_with_concurrency(
            2, tracked_task(1), failing_task(), tracked_task(2), return_exceptions=True
        )
        assert isinstance(results[1], ValueError)
        assert results[0] == 1
        assert results[2] == 2

        # Test cancellation
        # Create and start tasks immediately
        tasks = [asyncio.create_task(tracked_task(i, delay=1.0)) for i in range(3)]

        gather_task = asyncio.create_task(async_.gather_with_concurrency(2, *tasks))

        try:
            # Let tasks start
            await asyncio.sleep(0.1)
            # Cancel the gathering
            gather_task.cancel()

            with pytest.raises(asyncio.CancelledError):
                await gather_task
        finally:
            # Clean up any remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        # Verify proper cleanup after cancellation
        await asyncio.sleep(0.1)  # Let cleanup complete
        # Check that no tasks are still running
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                assert task.done(), "Task was not properly cleaned up"

    def test_to_sync(self) -> None:
        """Test to_sync decorator with comprehensive coverage.

        Tests:
        1. Basic async to sync conversion
        2. Error propagation
        3. Thread safety with multiple calls
        4. Resource cleanup
        5. Loop management
        6. Nested event loop prevention
        7. Exception context preservation
        8. State isolation between calls
        """
        results: list[int] = []
        errors: list[Exception] = []
        cleanup_called = threading.Event()

        @async_.to_sync
        async def async_func(x: int) -> int:
            await asyncio.sleep(0.1)
            return x * 2

        @async_.to_sync
        async def failing_async_func() -> None:
            await asyncio.sleep(0.1)
            raise ValueError("Async error")

        @async_.to_sync
        async def resource_func() -> None:
            try:
                await asyncio.sleep(0.1)
            finally:
                cleanup_called.set()

        @async_.to_sync
        async def nested_loop_func() -> None:
            # This should raise if we try to create a nested event loop
            await asyncio.sleep(0.1)
            # Create but don't run the coroutine to avoid "never awaited" warning
            coro = asyncio.sleep(0.1)
            try:
                asyncio.run(coro)  # This should fail
            finally:
                # Clean up the coroutine
                coro.close()

        # Test successful execution
        result = async_func(21)
        assert result == 42

        # Test error handling with proper context
        with pytest.raises(ValueError) as exc_info:
            failing_async_func()
        assert "Async error" in str(exc_info.value)
        assert exc_info.type is ValueError

        # Test resource cleanup
        resource_func()
        assert cleanup_called.is_set(), "Cleanup was not called"

        # Test nested event loop prevention
        with pytest.raises(RuntimeError) as exc_info:
            nested_loop_func()
        assert "cannot be called from a running event loop" in str(exc_info.value)

        # Test thread safety with multiple calls
        def worker(x: int) -> None:
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = async_func(x)
                    results.append(result)
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify thread safety
        assert len(results) == 5, f"Expected 5 results, got {len(results)}, errors: {errors}"
        assert sorted(results) == [0, 2, 4, 6, 8]
        assert not errors, f"Unexpected errors: {errors}"

        # Test state isolation
        state_values: list[int] = []

        @async_.to_sync
        async def stateful_func() -> None:
            # Each call should get its own isolated state
            local_val = 42
            await asyncio.sleep(0.1)
            state_values.append(local_val)

        # Run concurrent calls to check state isolation
        def state_worker() -> None:
            try:
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    stateful_func()
                finally:
                    loop.close()
                    asyncio.set_event_loop(None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=state_worker) for _ in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert all(val == 42 for val in state_values), "State was not properly isolated"
        assert len(state_values) == 3, f"Expected 3 state values, got {len(state_values)}"

        # Test error context preservation
        def get_error_context() -> None:
            try:
                raise RuntimeError("Original error")
            except RuntimeError:
                failing_async_func()

        with pytest.raises(ValueError) as exc_info:
            get_error_context()
        assert "Async error" in str(exc_info.value)
        assert exc_info.type is ValueError

    async def test_fire_coroutine_threadsafe(self) -> None:
        """Test fire_coroutine_threadsafe functionality with comprehensive coverage.

        Tests:
        1. Basic coroutine execution
        2. Thread safety with multiple concurrent calls
        3. Error propagation
        4. Resource cleanup
        5. State verification
        6. Cancellation handling
        """
        loop = asyncio.get_running_loop()
        events: list[asyncio.Event] = [asyncio.Event() for _ in range(5)]
        errors: list[Exception] = []
        cleanup_called = threading.Event()

        # Test successful execution with cleanup
        async def success_coro() -> None:
            try:
                await asyncio.sleep(0.1)
                events[0].set()
            finally:
                cleanup_called.set()

        async_.fire_coroutine_threadsafe(success_coro(), loop)
        await asyncio.wait_for(events[0].wait(), timeout=1)
        assert events[0].is_set()
        assert cleanup_called.is_set()

        # Test error propagation
        error_event = asyncio.Event()

        async def failing_coro() -> None:
            try:
                await asyncio.sleep(0.1)
                raise ValueError("Coroutine error")
            except Exception as e:
                errors.append(e)
                error_event.set()
                raise

        async_.fire_coroutine_threadsafe(failing_coro(), loop)
        await asyncio.wait_for(error_event.wait(), timeout=1)
        assert len(errors) == 1
        assert isinstance(errors[0], ValueError)
        assert str(errors[0]) == "Coroutine error"

        # Test thread safety with multiple concurrent calls
        async def concurrent_coro(idx: int) -> None:
            await asyncio.sleep(0.1)
            events[idx].set()

        def thread_worker(idx: int) -> None:
            try:
                coro = concurrent_coro(idx)
                async_.fire_coroutine_threadsafe(coro, loop)
            except Exception as e:
                errors.append(e)

        # Run multiple threads
        threads = [
            threading.Thread(target=thread_worker, args=(i,))
            for i in range(1, 5)  # Use events[1] through events[4]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wait for all events with timeout
        await asyncio.wait_for(asyncio.gather(*(events[i].wait() for i in range(1, 5))), timeout=2)

        # Verify all events were set
        assert all(event.is_set() for event in events[1:5]), "Not all coroutines completed"
        assert len(errors) == 1, f"Unexpected errors occurred: {errors}"  # Only the one from failing_coro

        # Test invalid input
        with pytest.raises(TypeError, match="A coroutine object is required"):
            async_.fire_coroutine_threadsafe(lambda: None, loop)  # type: ignore

        # Test cancellation
        cancelled = threading.Event()

        async def long_coro() -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        # Create a task we can cancel
        task = asyncio.create_task(long_coro())
        async_.fire_coroutine_threadsafe(task, loop)
        await asyncio.sleep(0.1)  # Let coroutine start
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        assert cancelled.is_set(), "Cancellation was not handled"
        assert task.cancelled(), "Task was not cancelled"

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
