#!/usr/bin/env python3

"""Script to merge template variables from multiple sources into a prompt template."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

import structlog

logger = structlog.get_logger(__name__)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load variables from a YAML configuration file.

    Args:
        config_path: Path to the YAML config file

    Returns:
        Dict containing the loaded variables

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in config file: {e}")
        raise


def load_file_contents(target_file: Path) -> str:
    """Load file contents using UV run.

    Args:
        target_file: Path to the target file to load

    Returns:
        Contents of the file as a string

    Raises:
        subprocess.CalledProcessError: If UV command fails
        FileNotFoundError: If target file doesn't exist
    """
    if not target_file.exists():
        raise FileNotFoundError(f"Target file not found: {target_file}")

    try:
        result = subprocess.run(
            ["uv", "run", str(target_file)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running UV command: {e}")
        raise


def load_template(template_path: Path) -> str:
    """Load the template file.

    Args:
        template_path: Path to the template file

    Returns:
        Template contents as a string

    Raises:
        FileNotFoundError: If template file doesn't exist
    """
    try:
        with open(template_path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Template file not found: {template_path}")
        raise


def merge_variables(
    template: str,
    config_vars: dict[str, Any],
    file_contents: str | None = None,
) -> str:
    """Merge all variables into the template.

    Args:
        template: The template string
        config_vars: Variables from YAML config
        file_contents: Optional file contents to merge

    Returns:
        The final prompt with all variables merged

    Raises:
        KeyError: If required variable is missing from config
    """
    variables = config_vars.copy()
    if file_contents is not None:
        variables["file_contents"] = file_contents

    try:
        return template.format(**variables)
    except KeyError as e:
        logger.error(f"Missing required variable in config: {e}")
        raise


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Generate a prompt by merging template variables from multiple sources"
    )
    parser.add_argument("template", type=Path, help="Path to template file")
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    parser.add_argument("target_file", type=Path, nargs="?", help="Optional target file to process")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file (defaults to stdout)",
    )

    args = parser.parse_args()

    try:
        # Load template and config
        template = load_template(args.template)
        config_vars = load_yaml_config(args.config)

        # Load target file if specified
        file_contents = None
        if args.target_file:
            file_contents = load_file_contents(args.target_file)

        # Merge variables
        result = merge_variables(template, config_vars, file_contents)

        # Output result
        if args.output:
            args.output.write_text(result)
            logger.info(f"Wrote output to {args.output}")
        else:
            print(result)

    except Exception as e:
        logger.error(f"Error generating prompt: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
