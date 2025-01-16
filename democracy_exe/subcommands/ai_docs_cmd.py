"""AI documentation generation commands"""
from __future__ import annotations

import asyncio
import os
import sys

from pathlib import Path
from typing import Annotated

import structlog
import typer

from rich import print as rprint

from democracy_exe.asynctyper import AsyncTyperImproved
from democracy_exe.utils.ai_docs_utils.extract_repo import extract_local_directory
from democracy_exe.utils.ai_docs_utils.generate_docs import agenerate_docs_from_local_repo


logger = structlog.get_logger(__name__)

APP = AsyncTyperImproved(help="ai docs command")


@APP.command("generate")
def cli_generate_docs(
    directory_path: Annotated[str, typer.Argument(help="Path to local repository directory")]
) -> None:
    """Generate documentation for a local repository.

    Args:
        directory_path: Path to the local repository directory
    """
    path = Path(directory_path)
    if not path.exists():
        rprint(f"[red]Directory does not exist: {directory_path}[/red]")
        raise typer.Exit(1)
    if not path.is_dir():
        rprint(f"[red]Not a directory: {directory_path}[/red]")
        raise typer.Exit(1)

    try:
        # Run in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(agenerate_docs_from_local_repo(str(path)))
        loop.close()

        rprint(f"[green]Documentation generated for {directory_path}[/green]")
    except Exception as ex:
        logger.error("Error generating documentation", error=str(ex))
        rprint(f"[red]Error: {ex!s}[/red]")
        raise typer.Exit(1)


@APP.command("extract")
def cli_extract_repo(
    directory_path: Annotated[str, typer.Argument(help="Path to local repository directory")]
) -> None:
    """Extract repository contents to a text file.

    Args:
        directory_path: Path to the local repository directory
    """
    path = Path(directory_path)
    if not path.exists():
        rprint(f"[red]Directory does not exist: {directory_path}[/red]")
        raise typer.Exit(1)
    if not path.is_dir():
        rprint(f"[red]Not a directory: {directory_path}[/red]")
        raise typer.Exit(1)

    try:
        output_file = extract_local_directory(str(path))
        rprint(f"[green]Repository extracted to {output_file}[/green]")
    except Exception as ex:
        logger.error("Error extracting repository", error=str(ex))
        rprint(f"[red]Error: {ex!s}[/red]")
        raise typer.Exit(1)


@APP.command("generate-async")
async def aio_cli_generate_docs(
    directory_path: Annotated[str, typer.Argument(help="Path to local repository directory")]
) -> None:
    """Generate documentation for a local repository asynchronously.

    Args:
        directory_path: Path to the local repository directory
    """
    path = Path(directory_path)
    if not path.exists():
        rprint(f"[red]Directory does not exist: {directory_path}[/red]")
        raise typer.Exit(1)
    if not path.is_dir():
        rprint(f"[red]Not a directory: {directory_path}[/red]")
        raise typer.Exit(1)

    try:
        await agenerate_docs_from_local_repo(str(path))
        rprint(f"[green]Documentation generated for {directory_path}[/green]")
    except Exception as ex:
        logger.error("Error generating documentation", error=str(ex))
        rprint(f"[red]Error: {ex!s}[/red]")
        raise typer.Exit(1)


@APP.command("extract-async")
async def aio_cli_extract_repo(
    directory_path: Annotated[str, typer.Argument(help="Path to local repository directory")]
) -> None:
    """Extract repository contents to a text file asynchronously.

    Args:
        directory_path: Path to the local repository directory
    """
    path = Path(directory_path)
    if not path.exists():
        rprint(f"[red]Directory does not exist: {directory_path}[/red]")
        raise typer.Exit(1)
    if not path.is_dir():
        rprint(f"[red]Not a directory: {directory_path}[/red]")
        raise typer.Exit(1)

    try:
        output_file = extract_local_directory(str(path))  # This function isn't async
        rprint(f"[green]Repository extracted to {output_file}[/green]")
    except Exception as ex:
        logger.error("Error extracting repository", error=str(ex))
        rprint(f"[red]Error: {ex!s}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    APP()
