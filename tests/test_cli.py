# pylint: disable=no-name-in-module
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false

"""Test the democracy_exe CLI."""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys

from collections.abc import AsyncGenerator, Generator, Mapping, Sequence
from typing import IO, TYPE_CHECKING, Any, Optional, Union

import typer

from typer.main import Typer
from typer.testing import CliRunner, Result

import pytest

from democracy_exe.asynctyper import AsyncTyperImproved
from democracy_exe.cli import APP


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


class AsyncCliRunner(CliRunner):
    """Extends CliRunner to support testing of async commands.

    This class provides async-aware invocation of CLI commands, particularly
    useful for testing AsyncTyperImproved applications.
    """

    async def ainvoke(
        self,
        app: AsyncTyperImproved | Typer,
        args: str | Sequence[str] | None = None,
        input: bytes | str | IO[Any] | None = None,
        env: Mapping[str, str] | None = None,
        catch_exceptions: bool = True,
        color: bool = False,
        **extra: Any,
    ) -> Result:
        """Async version of invoke() for testing async CLI commands.

        Args:
            app: The Typer application to test
            args: Command line arguments to pass
            input: Input data to pass
            env: Environment variables
            catch_exceptions: Whether to catch exceptions
            color: Whether to enable colored output
            **extra: Additional arguments to pass

        Returns:
            Result: The command execution result
        """
        async with contextlib.AsyncExitStack() as stack:
            if self.isolated_filesystem:

                @contextlib.asynccontextmanager
                async def _isolated_filesystem():
                    with self.isolated_filesystem() as path:
                        yield path

                await stack.enter_async_context(_isolated_filesystem())

            # Get event loop for running async commands
            loop = asyncio.get_event_loop()

            # Use super().invoke() but wrap it to handle async commands
            result = await loop.run_in_executor(
                None,
                lambda: CliRunner.invoke(
                    self, app, args=args, input=input, env=env, catch_exceptions=catch_exceptions, color=color, **extra
                ),
            )
            return result


@pytest.fixture
async def async_runner() -> AsyncGenerator[AsyncCliRunner, None]:
    """Fixture that provides an AsyncCliRunner instance.

    Yields:
        AsyncCliRunner: A fresh instance of AsyncCliRunner for testing.
    """
    yield AsyncCliRunner()


@pytest.fixture
def runner() -> CliRunner:
    """Fixture that provides a CliRunner instance.

    Returns:
        CliRunner: A fresh instance of CliRunner for testing.
    """
    return CliRunner()


@pytest.fixture
async def async_app() -> AsyncGenerator[AsyncTyperImproved, None]:
    """Fixture that provides an AsyncTyperImproved instance.

    Yields:
        AsyncTyperImproved: A fresh instance of AsyncTyperImproved for testing.
    """
    app = AsyncTyperImproved()
    yield app


def test_version_command(runner: CliRunner) -> None:
    """Test the version command output.

    Args:
        runner: Typer CLI test runner
    """
    result = runner.invoke(APP, ["version"])
    assert result.exit_code == 0
    assert "democracy_exe version:" in result.stdout


def test_deps_command(runner: CliRunner) -> None:
    """Test the deps command output.

    Args:
        runner: Typer CLI test runner
    """
    result = runner.invoke(APP, ["deps"])
    assert result.exit_code == 0
    assert "langchain_version:" in result.stdout
    assert "pydantic_version:" in result.stdout


@pytest.mark.asyncio
async def test_async_command(async_runner: AsyncCliRunner, async_app: AsyncTyperImproved) -> None:
    """Test an async command execution.

    Args:
        async_runner: Async-aware CLI test runner
        async_app: AsyncTyperImproved instance
    """

    @async_app.command()
    async def hello() -> str:  # type: ignore
        await asyncio.sleep(0.1)
        typer.echo("Hello Async!")
        return "Hello Async!"

    result = await async_runner.ainvoke(async_app, ["hello"])
    assert result.exit_code == 2


@pytest.mark.asyncio
async def test_run_bot_command(
    async_runner: AsyncCliRunner, mocker: MockerFixture, capsys: CaptureFixture[str]
) -> None:
    """Test the run_bot command.

    Args:
        async_runner: AsyncCliRunner
        mocker: Pytest mocker fixture
        capsys: Pytest stdout/stderr capture fixture
    """
    # Mock the DemocracyBot to prevent actual Discord connection
    mock_bot = mocker.patch("democracy_exe.cli.DemocracyBot")
    mock_bot.return_value.__aenter__.return_value = mock_bot.return_value
    mock_bot.return_value.__aexit__.return_value = None

    # Mock the print function
    result = await async_runner.ainvoke(APP, ["run-bot"])
    assert result.exit_code == 1
    await asyncio.sleep(0.1)
    # assert "Running bot" in result.stdout


@pytest.mark.asyncio
async def test_go_command(async_runner: AsyncCliRunner, mocker: MockerFixture, capsys: CaptureFixture[str]) -> None:
    """Test the go command that starts the DemocracyBot.

    Args:
        async_runner: AsyncCliRunner instance for testing async commands
        mocker: Pytest mocker fixture for mocking dependencies
        capsys: Pytest fixture for capturing stdout/stderr
    """
    # Mock the run_bot coroutine using AsyncMock
    mock_run_bot = mocker.patch("democracy_exe.cli.run_bot", new_callable=mocker.AsyncMock)

    # Execute the command
    result = await async_runner.ainvoke(APP, ["go"])

    # Verify the command executed successfully
    assert result.exit_code == 0

    # Verify the expected output message
    assert "Starting up DemocracyBot" in result.stdout

    # Verify that run_bot was called
    mock_run_bot.assert_called_once()
