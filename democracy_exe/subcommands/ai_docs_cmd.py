"""AI documentation generation commands"""

from __future__ import annotations

import asyncio
import os
import sys

from pathlib import Path
from typing import Optional

import structlog
import typer

from rich import print as rprint

from democracy_exe.asynctyper import AsyncTyperImproved
from democracy_exe.utils.ai_docs_utils.extract_repo import extract_local_directory
from democracy_exe.utils.ai_docs_utils.generate_docs import generate_docs_from_local_repo


logger = structlog.get_logger(__name__)

APP = AsyncTyperImproved(help="ai docs command")


@APP.command("generate")
def cli_generate_docs(
    directory_path: str = typer.Argument(..., help="Path to local repository directory")
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
        generate_docs_from_local_repo(directory_path)
        rprint(f"[green]Documentation generated for {directory_path}[/green]")
        return 0
    except Exception as ex:
        logger.error("Error generating documentation", error=str(ex))
        rprint(f"[red]Error: {ex!s}[/red]")
        return 1


@APP.command("extract")
def cli_extract_repo(
    directory_path: str = typer.Argument(..., help="Path to local repository directory")
) -> None:
    """Extract repository contents to a text file.

    Args:
        directory_path: Path to the local repository directory
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            rprint(f"[red]Directory does not exist: {directory_path}[/red]")
            raise typer.Exit(1)
        if not path.is_dir():
            rprint(f"[red]Not a directory: {directory_path}[/red]")
            raise typer.Exit(1)

        output_file = extract_local_directory(directory_path)
        rprint(f"[green]Repository extracted to {output_file}[/green]")

    except Exception as ex:
        logger.error("Error extracting repository", error=str(ex))
        rprint(f"[red]Error: {ex!s}[/red]")
        raise typer.Exit(1)


@APP.command("generate-async")
def aio_cli_generate_docs(
    directory_path: str = typer.Argument(..., help="Path to local repository directory")
) -> None:
    """Generate documentation for a local repository asynchronously.

    Args:
        directory_path: Path to the local repository directory
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            rprint(f"[red]Directory does not exist: {directory_path}[/red]")
            raise typer.Exit(1)
        if not path.is_dir():
            rprint(f"[red]Not a directory: {directory_path}[/red]")
            raise typer.Exit(1)

        asyncio.run(generate_docs_from_local_repo(directory_path))
        rprint(f"[green]Documentation generated for {directory_path}[/green]")

    except Exception as ex:
        logger.error("Error generating documentation", error=str(ex))
        rprint(f"[red]Error: {ex!s}[/red]")
        raise typer.Exit(1)


@APP.command("extract-async")
def aio_cli_extract_repo(
    directory_path: str = typer.Argument(..., help="Path to local repository directory")
) -> None:
    """Extract repository contents to a text file asynchronously.

    Args:
        directory_path: Path to the local repository directory
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            rprint(f"[red]Directory does not exist: {directory_path}[/red]")
            raise typer.Exit(1)
        if not path.is_dir():
            rprint(f"[red]Not a directory: {directory_path}[/red]")
            raise typer.Exit(1)

        output_file = asyncio.run(extract_local_directory(directory_path))
        rprint(f"[green]Repository extracted to {output_file}[/green]")

    except Exception as ex:
        logger.error("Error extracting repository", error=str(ex))
        rprint(f"[red]Error: {ex!s}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    APP()
