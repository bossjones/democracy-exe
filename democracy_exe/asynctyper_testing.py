from __future__ import annotations

import asyncio

from collections.abc import Mapping, Sequence
from typing import IO, Any, Optional, Union

from click.testing import CliRunner as ClickCliRunner
from click.testing import Result
from typer.main import get_command as _get_command

from democracy_exe.asynctyper import AsyncTyper, AsyncTyperImproved


class AsyncCliRunner(ClickCliRunner):
    """A test runner for AsyncTyper CLI applications.

    Extends CliRunner to handle async commands in AsyncTyper and AsyncTyperImproved applications.
    """

    def invoke(
        self,
        app: AsyncTyper | AsyncTyperImproved,
        args: str | Sequence[str] | None = None,
        input: bytes | str | IO[Any] | None = None,
        env: Mapping[str, str] | None = None,
        catch_exceptions: bool = True,
        color: bool = False,
        **extra: Any,
    ) -> Result:
        """Invoke the CLI application in an async context.

        Args:
            app: The AsyncTyper or AsyncTyperImproved application to test
            args: Command line arguments to pass
            input: Input data to provide
            env: Environment variables
            catch_exceptions: Whether to catch exceptions
            color: Whether to include ANSI color codes in output
            **extra: Additional arguments to pass to the command

        Returns:
            Result: The result of running the command
        """
        use_cli = _get_command(app)

        # If we're already in an event loop, just run normally
        try:
            loop = asyncio.get_running_loop()
            return super().invoke(
                use_cli,
                args=args,
                input=input,
                env=env,
                catch_exceptions=catch_exceptions,
                color=color,
                **extra,
            )
        except RuntimeError:
            # No event loop running, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return super().invoke(
                use_cli,
                args=args,
                input=input,
                env=env,
                catch_exceptions=catch_exceptions,
                color=color,
                **extra,
            )
        finally:
            loop.close()
