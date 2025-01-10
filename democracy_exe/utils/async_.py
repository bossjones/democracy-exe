"""Asyncio utilities with improved thread safety and resource management."""
from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import threading
import time
import typing
import weakref

from asyncio import Semaphore, coroutines, ensure_future, gather, get_running_loop
from asyncio.events import AbstractEventLoop
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from traceback import extract_stack
from typing import Any, List, Optional, TypeVar

import structlog

from codetiming import Timer


logger = structlog.get_logger(__name__)


_SHUTDOWN_RUN_CALLBACK_THREADSAFE = "_shutdown_run_callback_threadsafe"
_thread_pools: weakref.WeakSet = weakref.WeakSet()
_global_cleanup_lock = threading.Lock()
_global_shutdown = False

T = TypeVar("T")


def register_thread_pool(pool: ThreadPoolExecutor) -> None:
    """Register a thread pool for proper cleanup during shutdown.

    Args:
        pool: ThreadPoolExecutor to register
    """
    with _global_cleanup_lock:
        if not _global_shutdown:
            _thread_pools.add(pool)


def cleanup_thread_pools() -> None:
    """Clean up all registered thread pools."""
    global _global_shutdown  # pylint: disable=global-statement

    with _global_cleanup_lock:
        _global_shutdown = True
        for pool in _thread_pools:
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Error shutting down thread pool", error=str(e))


class AsyncContextManager:
    """Base class for async context managers with proper cleanup."""

    async def __aenter__(self) -> Any:
        """Enter the async context.

        Returns:
            Self instance
        """
        return self

    async def __aexit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> None:
        """Exit the async context.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        pass


class ThreadSafeEvent:
    """A thread-safe event wrapper for coordinating between threads and coroutines.

    This class provides a thread-safe wrapper around asyncio.Event that can be used
    to coordinate between threads and coroutines. The event loop is stored at creation
    time and used for all operations to ensure thread safety.
    """

    def __init__(self) -> None:
        """Initialize the event with the current event loop."""
        self._event = asyncio.Event()
        self._lock = threading.Lock()
        self._is_set = False
        self._closed = False
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("ThreadSafeEvent must be created from an async context")

    def set(self) -> None:
        """Set the event in a thread-safe manner."""
        with self._lock:
            if self._closed:
                return
            self._is_set = True
            if self._event and not self._event.is_set():
                try:
                    self._loop.call_soon_threadsafe(self._event.set)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("Error setting event", error=str(e))

    def clear(self) -> None:
        """Clear the event in a thread-safe manner."""
        with self._lock:
            if self._closed:
                return
            self._is_set = False
            if self._event and self._event.is_set():
                try:
                    self._loop.call_soon_threadsafe(self._event.clear)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("Error clearing event", error=str(e))

    async def wait(self) -> None:
        """Wait for the event to be set.

        Raises:
            RuntimeError: If the event is closed
        """
        if self._closed:
            raise RuntimeError("Event is closed")
        if not self._is_set:
            await self._event.wait()

    def close(self) -> None:
        """Close the event and cleanup resources."""
        with self._lock:
            self._closed = True
            self._event = None
            self._loop = None  # type: ignore


class AsyncSemaphore(AsyncContextManager):
    """Thread-safe semaphore that can be used across async and sync code."""

    def __init__(self, value: int = 1) -> None:
        """Initialize the semaphore.

        Args:
            value: Initial semaphore value

        Raises:
            ValueError: If value is less than 1
        """
        super().__init__()
        if value < 1:
            raise ValueError("Semaphore initial value must be >= 1")
        self._value = value  # Store initial value
        self._semaphore = Semaphore(value)
        self._lock = threading.Lock()
        self._count = value  # Current available count
        self._waiters: list[asyncio.Future] = []
        self._closed = False
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("AsyncSemaphore must be created from an async context")

    async def acquire(self) -> bool:
        """Acquire the semaphore.

        Returns:
            bool: True if acquired, False if closed

        Raises:
            RuntimeError: If semaphore is closed
        """
        if self._closed:
            raise RuntimeError("Semaphore is closed")

        # First acquire the underlying semaphore
        await self._semaphore.acquire()

        with self._lock:
            if self._count > 0:
                self._count -= 1
                return True

            # Create a waiter before releasing the lock
            waiter: asyncio.Future = asyncio.Future()
            self._waiters.append(waiter)

        try:
            # Wait for our turn
            await waiter
            self._count -= 1  # Decrement count after being woken up
            return True
        except Exception:  # pylint: disable=broad-except
            with self._lock:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self._semaphore.release()  # Release semaphore if we failed
            raise

    def release(self) -> None:
        """Release the semaphore.

        Raises:
            RuntimeError: If semaphore is closed
        """
        if self._closed:
            raise RuntimeError("Semaphore is closed")

        with self._lock:
            # Release the underlying semaphore first
            try:
                self._loop.call_soon_threadsafe(self._semaphore.release)
            except Exception as e:
                logger.error("Error releasing semaphore", error=str(e))
                raise

            # Update count and wake waiters
            self._count += 1
            if self._waiters:
                waiter = self._waiters.pop(0)
                if not waiter.done():
                    self._loop.call_soon_threadsafe(waiter.set_result, None)

    def close(self) -> None:
        """Close the semaphore and cleanup resources."""
        with self._lock:
            self._closed = True
            # Wake up all waiters with an error
            for waiter in self._waiters:
                if not waiter.done():
                    self._loop.call_soon_threadsafe(
                        waiter.set_exception, RuntimeError("Semaphore is closed")
                    )
            self._waiters.clear()
            # Release all held resources
            while self._count < self._value:
                self._loop.call_soon_threadsafe(self._semaphore.release)
                self._count += 1

    async def __aenter__(self) -> AsyncSemaphore:
        """Enter async context and acquire semaphore.

        Returns:
            Self instance

        Raises:
            RuntimeError: If semaphore is closed
        """
        await self.acquire()
        return self

    async def __aexit__(
        self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        """Exit async context and release semaphore.

        Args:
            exc_type: Exception type if an error occurred
            exc_val: Exception value if an error occurred
            exc_tb: Exception traceback if an error occurred
        """
        self.release()


class ThreadPoolManager:
    """Manager for thread pools with proper lifecycle management."""

    def __init__(self) -> None:
        self._pools: weakref.WeakSet = weakref.WeakSet()
        self._lock = threading.Lock()
        self._closed = False

    def create_pool(self, *args, **kwargs) -> ThreadPoolExecutor:
        """Create and register a new thread pool.

        Args:
            *args: Positional arguments for ThreadPoolExecutor
            **kwargs: Keyword arguments for ThreadPoolExecutor

        Returns:
            ThreadPoolExecutor: New thread pool instance

        Raises:
            RuntimeError: If manager is closed
        """
        if self._closed:
            raise RuntimeError("ThreadPoolManager is closed")

        pool = ThreadPoolExecutor(*args, **kwargs)
        with self._lock:
            self._pools.add(pool)
        return pool

    def shutdown_all(self, wait: bool = True) -> None:
        """Shutdown all managed thread pools.

        Args:
            wait: Whether to wait for thread pools to finish
        """
        with self._lock:
            self._closed = True
            for pool in self._pools:
                try:
                    pool.shutdown(wait=wait, cancel_futures=True)
                except Exception as e:  # pylint: disable=broad-except
                    logger.error("Error shutting down thread pool", error=str(e))
            self._pools.clear()


# Global thread pool manager
thread_pool_manager = ThreadPoolManager()


async def gather_with_concurrency(limit: int, *tasks: Any, return_exceptions: bool = False) -> Any:
    """
    Wrap asyncio.gather to limit the number of concurrent tasks with improved error handling.

    Args:
        limit: Maximum number of concurrent tasks
        *tasks: Tasks to gather
        return_exceptions: Whether to return exceptions rather than raising them

    Returns:
        Results from the gathered tasks

    Raises:
        Exception: If a task fails and return_exceptions is False
        ValueError: If limit is less than 1
    """
    if limit < 1:
        raise ValueError("Concurrency limit must be >= 1")

    semaphore = AsyncSemaphore(limit)

    async def sem_task(task: Awaitable[Any]) -> Any:
        """Run a task with semaphore protection."""
        try:
            async with semaphore:
                return await task
        except Exception as e:  # pylint: disable=broad-except
            if return_exceptions:
                return e
            raise
        finally:
            # Ensure semaphore is released even if task is cancelled
            if not semaphore._closed:  # pylint: disable=protected-access
                try:
                    semaphore.release()
                except RuntimeError:
                    pass

    try:
        return await gather(*(sem_task(task) for task in tasks), return_exceptions=return_exceptions)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error in gathered tasks", error=str(e))
        raise
    finally:
        semaphore.close()


def to_async(func: typing.Callable) -> typing.Callable:
    """
    Turn a sync function to async function using event loop with proper error handling.

    Args:
        func: Synchronous function to convert

    Returns:
        Async version of the function
    """

    @functools.wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

        pfunc = functools.partial(func, *args, **kwargs)
        try:
            return await loop.run_in_executor(executor, pfunc)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Error in async execution",
                function=func.__name__,
                error=str(e),
                args=args,
                kwargs=kwargs,
            )
            raise

    return run


def to_async_thread(fn):
    """
    Turn a sync function to async function using threads with proper cleanup.

    Args:
        fn: Synchronous function to convert

    Returns:
        Async version of the function
    """
    pool = thread_pool_manager.create_pool()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            future = pool.submit(fn, *args, **kwargs)
            return asyncio.wrap_future(future)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Error submitting to thread pool",
                function=fn.__name__,
                error=str(e),
                args=args,
                kwargs=kwargs,
            )
            raise

    return wrapper


def to_sync(fn):
    """
    Turn an async function to sync function with proper error handling.

    Args:
        fn: Async function to convert

    Returns:
        Sync version of the function
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(res)
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Error in sync execution",
                    function=fn.__name__,
                    error=str(e),
                    args=args,
                    kwargs=kwargs,
                )
                raise
        return res

    return wrapper


def force_async(fn):
    """
    Turn a sync function to async function using threads with proper resource management.

    Args:
        fn: Synchronous function to convert

    Returns:
        Async version of the function
    """
    pool = thread_pool_manager.create_pool()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            future = pool.submit(fn, *args, **kwargs)
            return asyncio.wrap_future(future)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "Error in forced async execution",
                function=fn.__name__,
                error=str(e),
                args=args,
                kwargs=kwargs,
            )
            raise

    return wrapper


def force_sync(fn):
    """
    Turn an async function to sync function with proper error handling.

    Args:
        fn: Async function to convert

    Returns:
        Sync version of the function
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            try:
                loop = asyncio.get_running_loop()
                return loop.run_until_complete(res)
            except Exception as e:  # pylint: disable=broad-except
                logger.error(
                    "Error in forced sync execution",
                    function=fn.__name__,
                    error=str(e),
                    args=args,
                    kwargs=kwargs,
                )
                raise
        return res

    return wrapper


def fire_coroutine_threadsafe(coro: Coroutine, loop: AbstractEventLoop) -> asyncio.Future:
    """
    Submit a coroutine object to a given event loop with improved safety.

    This method provides a way to run a coroutine on a different event loop
    and retrieve its result through the returned Future object.

    Args:
        coro: Coroutine to run
        loop: Event loop to run the coroutine in

    Returns:
        asyncio.Future: A Future object that will contain the result of the coroutine

    Raises:
        RuntimeError: If called from within the event loop
        TypeError: If coro is not a coroutine object
    """
    ident = loop.__dict__.get("_thread_ident")
    if ident is not None and ident == threading.get_ident():
        raise RuntimeError("Cannot be called from within the event loop")

    if not coroutines.iscoroutine(coro):
        raise TypeError(f"A coroutine object is required: {coro}")

    done_event = ThreadSafeEvent()
    future_ref: list[asyncio.Future | None] = [None]  # Use a list to store future reference
    future_ready = threading.Event()

    def callback() -> None:
        """Handle the firing of a coroutine with proper error handling."""
        try:
            future = ensure_future(coro, loop=loop)
            future_ref[0] = future  # Store future reference
            future.add_done_callback(lambda _: done_event.set())
            future_ready.set()  # Signal that future is ready

            def _on_cancel(fut: asyncio.Future) -> None:
                """Handle future cancellation."""
                if not fut.cancelled():
                    return
                # Get the task and cancel it
                task = asyncio.tasks._get_current_task(loop)  # type: ignore
                if task is not None:
                    task.cancel()

            future.add_done_callback(_on_cancel)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error in coroutine", error=str(e))
            done_event.set()
            future_ready.set()

    try:
        loop.call_soon_threadsafe(callback)
        # Wait for the future to be created
        if not future_ready.wait(timeout=0.1):
            logger.warning("Timeout waiting for future to be created")
        return future_ref[0] if future_ref[0] is not None else asyncio.Future(loop=loop)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error scheduling coroutine", error=str(e))
        raise


def run_callback_threadsafe(
    loop: AbstractEventLoop, callback: Callable[..., T], *args: Any
) -> concurrent.futures.Future[T]:  # pylint: disable=unsubscriptable-object
    """
    Submit a callback object to a given event loop with improved safety.

    Return a concurrent.futures.Future to access the result.

    Args:
        loop: Event loop to run the callback in
        callback: Function to call
        *args: Arguments to pass to the callback

    Returns:
        Future containing the callback result

    Raises:
        RuntimeError: If called from within the event loop or during shutdown
    """
    ident = loop.__dict__.get("_thread_ident")
    if ident is not None and ident == threading.get_ident():
        raise RuntimeError("Cannot be called from within the event loop")

    future: concurrent.futures.Future = concurrent.futures.Future()
    done_event = ThreadSafeEvent()

    def run_callback() -> None:
        """Run callback and store result with proper error handling."""
        try:
            result = callback(*args)
            future.set_result(result)
        except Exception as exc:  # pylint: disable=broad-except
            if future.set_running_or_notify_cancel():
                future.set_exception(exc)
            else:
                logger.warning("Exception on lost future", exc_info=True)
        finally:
            done_event.set()

    try:
        loop.call_soon_threadsafe(run_callback)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error scheduling callback", error=str(e))
        future.set_exception(e)
        return future

    if hasattr(loop, _SHUTDOWN_RUN_CALLBACK_THREADSAFE):
        raise RuntimeError("The event loop is in the process of shutting down")

    return future


def check_loop() -> None:
    """
    Warn if called inside the event loop and provide guidance.

    Raises:
        RuntimeError: If I/O is detected inside the event loop
    """
    try:
        get_running_loop()
        in_loop = True
    except RuntimeError:
        in_loop = False

    if not in_loop:
        return

    found_frame = None

    for frame in reversed(extract_stack()):
        for path in ("custom_components/", "homeassistant/components/"):
            try:
                index = frame.filename.index(path)
                found_frame = frame
                break
            except ValueError:
                continue

        if found_frame is not None:
            break

    # Did not source from integration? Hard error.
    if found_frame is None:
        raise RuntimeError(
            "Detected I/O inside the event loop. This is causing stability issues. Please report issue"
        )

    start = index + len(path)
    end = found_frame.filename.index("/", start)

    integration = found_frame.filename[start:end]

    if path == "custom_components/":
        extra = " to the custom component author"
    else:
        extra = ""

    logger.warning(
        "Detected I/O inside the event loop. This is causing stability issues. Please report issue%s for %s doing I/O at %s, line %s: %s",
        extra,
        integration,
        found_frame.filename[index:],
        found_frame.lineno,
        found_frame.line.strip(),
    )

    raise RuntimeError(
        f"I/O must be done in the executor; Use `await loop.run_in_executor()` "
        f"at {found_frame.filename[index:]}, line {found_frame.lineno}: {found_frame.line.strip()}"
    )


def protect_loop(func: Callable) -> Callable:
    """
    Protect function from running in event loop.

    Args:
        func: Function to protect

    Returns:
        Protected function that cannot run in the event loop
    """

    @functools.wraps(func)
    def protected_loop_func(*args, **kwargs):  # type: ignore
        check_loop()
        return func(*args, **kwargs)

    return protected_loop_func


def async_timed():
    """Decorator for timing async functions with proper cleanup."""

    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapped(*args, **kwargs) -> Any:
            logger.debug(f"Starting {func.__name__}", args=args, kwargs=kwargs)
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                end = time.time()
                total = end - start
                logger.debug(
                    f"Finished {func.__name__}",
                    duration=f"{total:.4f}s",
                    args=args,
                    kwargs=kwargs,
                )

        return wrapped

    return wrapper


def async_timer():
    """Decorator for timing async functions using Timer with proper cleanup."""

    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapped(*args, **kwargs) -> Any:
            timer = Timer(text=f"Task {func.__name__} elapsed time: {{:.1f}}")
            logger.debug(f"Starting {func.__name__}", args=args, kwargs=kwargs)
            timer.start()
            try:
                return await func(*args, **kwargs)
            finally:
                timer.stop()
                logger.debug(
                    f"Finished {func.__name__}",
                    duration=timer.last,
                    args=args,
                    kwargs=kwargs,
                )

        return wrapped

    return wrapper


def shutdown_run_callback_threadsafe(loop: AbstractEventLoop) -> None:
    """
    Call when run_callback_threadsafe should prevent creating new futures.

    We must finish all callbacks before the executor is shutdown
    or we can end up in a deadlock state where:

    `executor.result()` is waiting for its `._condition`
    and the executor shutdown is trying to `.join()` the
    executor thread.

    This function is considered irreversible and should only ever
    be called when Home Assistant is going to shutdown and
    python is going to exit.

    Args:
        loop: Event loop to mark as shutting down
    """
    setattr(loop, _SHUTDOWN_RUN_CALLBACK_THREADSAFE, True)
    cleanup_thread_pools()


# Register cleanup on module exit
import atexit


atexit.register(cleanup_thread_pools)
