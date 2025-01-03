"""Asyncio utilities."""
# SOURCE: https://github.com/home-assistant/core/blob/5983fac5c213acab799339c7baec43cf4300f196/homeassistant/util/async_.py

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import threading
import time
import typing

from asyncio import Semaphore, coroutines, ensure_future, gather, get_running_loop
from asyncio.events import AbstractEventLoop
from collections.abc import Awaitable, Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from traceback import extract_stack
from typing import Any, TypeVar

from codetiming import Timer
from loguru import logger as LOGGER


_SHUTDOWN_RUN_CALLBACK_THREADSAFE = "_shutdown_run_callback_threadsafe"

T = TypeVar("T")


# https://stackoverflow.com/questions/33128325/how-to-set-class-attribute-with-await-in-init
class aobject:
    """
    Inheriting this class allows you to define an async __init__.

    So you can create objects by doing something like `await MyClass(params)`
    """

    async def __new__(cls, *a, **kw):
        instance = super().__new__(cls)
        await instance.__init__(*a, **kw)
        return instance

    async def __init__(self):
        pass


def fire_coroutine_threadsafe(coro: Coroutine, loop: AbstractEventLoop) -> None:
    """
    Submit a coroutine object to a given event loop.

    This method does not provide a way to retrieve the result and
    is intended for fire-and-forget use. This reduces the
    work involved to fire the function on the loop.
    """
    ident = loop.__dict__.get("_thread_ident")
    if ident is not None and ident == threading.get_ident():
        raise RuntimeError("Cannot be called from within the event loop")

    if not coroutines.iscoroutine(coro):
        raise TypeError(f"A coroutine object is required: {coro}")

    def callback() -> None:
        """Handle the firing of a coroutine."""
        ensure_future(coro, loop=loop)

    loop.call_soon_threadsafe(callback)


def run_callback_threadsafe(
    loop: AbstractEventLoop, callback: Callable[..., T], *args: Any
) -> concurrent.futures.Future[T]:  # pylint: disable=unsubscriptable-object
    """
    Submit a callback object to a given event loop.

    Return a concurrent.futures.Future to access the result.
    """
    ident = loop.__dict__.get("_thread_ident")
    if ident is not None and ident == threading.get_ident():
        raise RuntimeError("Cannot be called from within the event loop")

    future: concurrent.futures.Future = concurrent.futures.Future()

    def run_callback() -> None:
        """Run callback and store result."""
        try:
            future.set_result(callback(*args))
        except Exception as exc:  # pylint: disable=broad-except
            if future.set_running_or_notify_cancel():
                future.set_exception(exc)
            else:
                LOGGER.warning("Exception on lost future: ", exc_info=True)

    loop.call_soon_threadsafe(run_callback)

    if hasattr(loop, _SHUTDOWN_RUN_CALLBACK_THREADSAFE):
        #
        # If the final `HomeAssistant.async_block_till_done` in
        # `HomeAssistant.async_stop` has already been called, the callback
        # will never run and, `future.result()` will block forever which
        # will prevent the thread running this code from shutting down which
        # will result in a deadlock when the main thread attempts to shutdown
        # the executor and `.join()` the thread running this code.
        #
        # To prevent this deadlock we do the following on shutdown:
        #
        # 1. Set the _SHUTDOWN_RUN_CALLBACK_THREADSAFE attr on this function
        #    by calling `shutdown_run_callback_threadsafe`
        # 2. Call `hass.async_block_till_done` at least once after shutdown
        #    to ensure all callbacks have run
        # 3. Raise an exception here to ensure `future.result()` can never be
        #    called and hit the deadlock since once `shutdown_run_callback_threadsafe`
        #    we cannot promise the callback will be executed.
        #
        raise RuntimeError("The event loop is in the process of shutting down.")

    LOGGER.complete()
    return future


def check_loop() -> None:
    """Warn if called inside the event loop."""
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
        raise RuntimeError("Detected I/O inside the event loop. This is causing stability issues. Please report issue")

    start = index + len(path)
    end = found_frame.filename.index("/", start)

    integration = found_frame.filename[start:end]

    if path == "custom_components/":
        extra = " to the custom component author"
    else:
        extra = ""

    LOGGER.warning(
        "Detected I/O inside the event loop. This is causing stability issues. Please report issue%s for %s doing I/O at %s, line %s: %s",
        extra,
        integration,
        found_frame.filename[index:],
        found_frame.lineno,
        found_frame.line.strip(),
    )
    LOGGER.complete()
    raise RuntimeError(
        f"I/O must be done in the executor; Use `await hass.async_add_executor_job()` "
        f"at {found_frame.filename[index:]}, line {found_frame.lineno}: {found_frame.line.strip()}"
    )


def protect_loop(func: Callable) -> Callable:
    """Protect function from running in event loop."""

    @functools.wraps(func)
    def protected_loop_func(*args, **kwargs):  # type: ignore
        check_loop()
        return func(*args, **kwargs)

    return protected_loop_func


async def gather_with_concurrency(limit: int, *tasks: Any, return_exceptions: bool = False) -> Any:
    """
    Wrap asyncio.gather to limit the number of concurrent tasks.

    From: https://stackoverflow.com/a/61478547/9127614
    """
    semaphore = Semaphore(limit)

    async def sem_task(task: Awaitable[Any]) -> Any:
        async with semaphore:
            return await task

    return await gather(*(sem_task(task) for task in tasks), return_exceptions=return_exceptions)


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
    """
    setattr(loop, _SHUTDOWN_RUN_CALLBACK_THREADSAFE, True)


# SOURCE: https://livebook.manning.com/book/concurrency-in-python-with-asyncio/chapter-2/v-10/123
def async_timed():
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapped(*args, **kwargs) -> Any:
            print(f"starting {func} with args {args} {kwargs}")
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                end = time.time()
                total = end - start
                print(f"finished {func} in {total:.4f} second(s)")

        return wrapped

    return wrapper


# Based on this originally, but modified to use Timer
# SOURCE: https://livebook.manning.com/book/concurrency-in-python-with-asyncio/chapter-2/v-10/123
def async_timer():
    def wrapper(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapped(*args, **kwargs) -> Any:
            timer = Timer(text=f"Task {__name__} elapsed time: {{:.1f}}")
            print(f"starting {func} with args {args} {kwargs}")
            timer.start()
            try:
                return await func(*args, **kwargs)
            finally:
                timer.stop()

        return wrapped

    return wrapper


#######################################
# START: https://gist.github.com/iedmrc/2fbddeb8ca8df25356d8acc3d297e955
# SOURCE: https://gist.github.com/iedmrc/2fbddeb8ca8df25356d8acc3d297e955
#######################################


def to_async(func: typing.Callable) -> typing.Callable:
    """Turns a sync function to async function using event loop"""

    @functools.wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)

    return run


def to_async_thread(fn):
    """Turns a sync function to async function using threads"""
    pool = ThreadPoolExecutor()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)  # make it awaitable

    return wrapper


def to_sync(fn):
    """Turns an async function to sync function"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            # return asyncio.get_event_loop().run_until_complete(res)
            # loop = asyncio.get_running_loop()
            if loop is None:
                loop = asyncio.get_event_loop()
            return loop.run_until_complete(res)
        return res

    return wrapper


def force_async(fn):
    """Turns a sync function to async function using threads"""
    import asyncio

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor()

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        return asyncio.wrap_future(future)  # make it awaitable

    return wrapper


def force_sync(fn):
    """Turn an async function to sync function"""
    import asyncio

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(res)
        return res

    return wrapper


#######################################
# END: https://gist.github.com/iedmrc/2fbddeb8ca8df25356d8acc3d297e955
# SOURCE: https://gist.github.com/iedmrc/2fbddeb8ca8df25356d8acc3d297e955
#######################################
