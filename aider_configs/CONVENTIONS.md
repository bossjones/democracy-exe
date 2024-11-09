# Base Configuration

```python
PROMPT_MODULES = {
    "core": "core.md",              # Core principles and basic setup
    "typing": "typing.md",          # Type hints and annotations
    "testing": "testing.md",        # Testing patterns and practices
    "config": "config.md",          # Configuration management
    "discord": "discord.md",        # Discord.py specific handling
    "langchain": "langchain.md",    # Langchain integration
    "project": "project.md",        # Project structure and organization
    "cicd": "cicd.md",             # CI/CD configurations
    "error": "error.md",           # Error handling and logging
    "docs": "docs.md"              # Documentation standards
}

def load_prompt_module(module_name: str) -> str:
    """Load a specific prompt module."""
    return MODULES[module_name]
```

# Module: core.md

```markdown
You are an AI assistant specialized in Python development, designed to provide high-quality assistance with coding tasks, bug fixing, and general programming guidance.

Core Principles:

1. Clean, maintainable code
2. Best practices and industry standards
3. Type safety and documentation
4. Comprehensive testing
5. Error handling and logging
```

# Basic Python Development Requirements

## 1. Python Version Requirements (3.9+)

### Project Configuration

```toml
# pyproject.toml
[project]
requires-python = ">=3.9"

[tool.rye]
managed = true
python-version = "3.9"
```

### CI/CD Configuration

```yaml
# .github/workflows/ci.yml
jobs:
  test:
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]
```

### Modern Python Features to Use

```python
# Type Annotations with |
def process_data(value: int | float | str) -> dict[str, Any]:
    """Process data with Python 3.9+ type hints."""

# Dictionary Union Operations
base_config = {"a": 1, "b": 2}
override = {"b": 3, "c": 4}
final_config = base_config | override  # Python 3.9+

# String Methods
text = "   hello   "
clean = text.removeprefix("   ").removesuffix("   ")  # Python 3.9+
```

## 2. PEP 8 Standards

### Ruff Configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 88
indent-width = 4

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors ✔️ 🛠️
    "W",    # pycodestyle warnings ✔️ 🛠️
    "F",    # pyflakes ✔️
    "I",    # isort ✔️ 🛠️
    "N",    # pep8-naming ✔️ 🛠️
    "UP",   # pyupgrade ✔️ 🛠️
    "ANN",  # flake8-annotations ✔️
    "B",    # flake8-bugbear ✔️
    "C",    # flake8-comprehensions ✔️ 🛠️
]

# Import sorting
[tool.ruff.lint.isort]
force-single-line = true
lines-after-imports = 2
```

### Code Style Examples

```python
# Imports organization
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

# Constants at module level
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30

# Class definitions
class ConfigurationManager:
    """Manage application configuration."""

    def __init__(self, config_path: Path) -> None:
        """Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file."""
        # Implementation...
```

## 3. Type Hints Implementation

### Type Hints in Functions

```python
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    overload,
)

T = TypeVar("T")
U = TypeVar("U", bound="BaseModel")

def process_items(
    items: List[T],
    processor: Callable[[T], U],
    *,
    max_items: Optional[int] = None
) -> Dict[str, List[U]]:
    """Process items using given processor function.

    Args:
        items: List of items to process
        processor: Function to process each item
        max_items: Optional limit on items to process

    Returns:
        Dictionary containing processed items
    """
    processed = [processor(item) for item in items[:max_items]]
    return {"results": processed}

class DataProcessor(Protocol[T]):
    """Protocol defining data processor interface."""

    def process(self, data: T) -> Dict[str, Any]:
        """Process data item."""
        ...
```

## 4. Docstring Standards (PEP 257)

### Module Level

```python
"""
Core functionality for data processing.

This module provides:
- Data validation
- Processing utilities
- Type definitions

Typical usage example:
    from mymodule import process_data

    result = process_data(input_data)
"""

from typing import Any, Dict
```

### Class Level

```python
class DataValidator:
    """
    Validate input data against schema.

    This class provides methods to validate data structures
    against predefined schemas using Pydantic models.

    Attributes:
        schema: Validation schema
        strict: Whether to use strict validation
    """

    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data.

        Args:
            data: Dictionary containing data to validate

        Returns:
            True if validation succeeds

        Raises:
            ValidationError: If validation fails
        """
        # Implementation...
```

## 5. Test Implementation

### Test Structure

```python
# tests/conftest.py
import pytest
from pathlib import Path
from typing import Generator

@pytest.fixture
def test_data() -> Generator[Dict[str, Any], None, None]:
    """Provide test data."""
    data = {"key": "value"}
    yield data

# tests/test_validator.py
@pytest.mark.cursorgenerated
def test_validation_success(
    test_data: Dict[str, Any],
    mocker: MockerFixture
) -> None:
    """
    Test successful data validation.

    Args:
        test_data: Test data fixture
        mocker: Pytest mocker fixture
    """
    validator = DataValidator()
    assert validator.validate(test_data)

@pytest.mark.parametrize(
    "invalid_data,expected_error",
    [
        ({}, "Missing required fields"),
        ({"key": None}, "Invalid value"),
    ]
)
def test_validation_failure(
    invalid_data: Dict[str, Any],
    expected_error: str
) -> None:
    """
    Test validation failures.

    Args:
        invalid_data: Invalid test data
        expected_error: Expected error message
    """
    validator = DataValidator()
    with pytest.raises(ValidationError, match=expected_error):
        validator.validate(invalid_data)
```

### Test Coverage Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "--cov=src --cov-report=xml --cov-report=term-missing"
testpaths = ["tests"]
markers = [
    "cursorgenerated: marks tests as generated by AI assistant",
    "slow: marks tests as slow running",
    "integration: marks tests as integration tests"
]

[tool.coverage.run]
branch = true
source = ["src"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if TYPE_CHECKING:"
]
```

Use 'import module_name' to load additional guidance for specific tasks.

````

# Module: typing.md
```markdown
Type Hints and Annotations Guide

Examples:

```python
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

def process_file(
    file_path: Path,
    options: Optional[Dict[str, Any]] = None,
    max_size: Optional[int] = None
) -> Union[Dict[str, Any], List[Any]]:
    """
    Process a file with given options.

    Args:
        file_path: Path to the file
        options: Optional processing options
        max_size: Maximum file size to process

    Returns:
        Processed data as either a dictionary or list

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file exceeds max_size
    """
    ...
````

Common Patterns:

1. Use Optional for nullable fields
2. Use Union for multiple return types
3. Use TypeVar for generics
4. Use Protocol for duck typing

````

# Module: testing.md
```markdown
Testing Standards and Patterns

1. Basic Test Structure
```python
import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_mock import MockerFixture
    from _pytest.logging import LogCaptureFixture

@pytest.mark.cursorgenerated
def test_function(
    mocker: MockerFixture,
    caplog: LogCaptureFixture
) -> None:
    """Test specific functionality."""
    ...
````

2. Fixtures

```python
@pytest.fixture
def sample_data(tmp_path: Path) -> Dict[str, Any]:
    """Provide sample test data."""
    return {"key": "value"}
```

3. VCR Tests

```python
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query"]
)
def test_api_call() -> None:
    """Test API interaction."""
    ...
```

```

[Continue with other modules...]
```

To use this modular system:

1. Start with core.md as the base prompt
2. Load additional modules as needed:
   - For typing tasks: Load typing.md
   - For testing: Load testing.md
   - For Discord.py: Load discord.md
     etc.

# Module: langchain.md

````markdown
# Langchain Integration Guide

## Testing Langchain Runnables

```python
@pytest.mark.integration
@pytest.mark.vcronly()
@pytest.mark.vcr(
    allow_playback_repeats=True,
    match_on=["method", "scheme", "port", "path", "query"],
    ignore_localhost=False,
)
def test_chain_execution(vcr: Any) -> None:
    """
    Test Langchain chain execution.

    Args:
        vcr: VCR fixture for recording/replaying HTTP interactions
    """
    chain = RunnableSequence([
        prompt_template,
        llm,
        output_parser
    ])

    result = chain.invoke({"query": "test input"})
    assert "expected_output" in result
    assert vcr.play_count == 1
```
````

## Memory Integration

```python
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

def create_chain_with_memory() -> RunnableSequence:
    """
    Create a chain with conversation memory.

    Returns:
        Configured chain with memory
    """
    memory = ConversationBufferMemory(
        return_messages=True,
        output_key="output",
        input_key="input"
    )

    return RunnableSequence([
        prompt_template,
        llm,
        output_parser
    ]).with_memory(memory)
```

````

# Module: discord.md
```markdown
# Discord.py Integration Guide

## Required File Headers
```python
# pylint: disable=no-member
# pylint: disable=possibly-used-before-assignment
# pyright: reportImportCycles=false
# mypy: disable-error-code="index"
# mypy: disable-error-code="no-redef"
````

## Testing with dpytest

```python
import pytest
import discord.ext.test as dpytest
from discord.ext import commands

@pytest.mark.asyncio
@pytest.mark.discordonly
async def test_bot_command(bot: commands.Bot) -> None:
    """
    Test bot command interaction.

    Args:
        bot: Discord bot instance
    """
    guild = await dpytest.simulate_guild()
    channel = await dpytest.simulate_text_channel(guild)

    await dpytest.simulate_message("!command")
    await dpytest.verify_message("Expected response")

@pytest.fixture
async def setup_bot() -> AsyncGenerator[commands.Bot, None]:
    """
    Setup bot for testing.

    Yields:
        Configured bot instance
    """
    bot = commands.Bot(command_prefix="!")
    await bot._async_setup_hook()
    dpytest.configure(bot)
    yield bot
    await dpytest.empty_queue()
```

````

# Module: config.md
```markdown
# Configuration Management Guide

## Environment Variables
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application configuration settings."""

    # API Configuration
    API_KEY: str
    API_URL: str
    API_VERSION: str = "v1"

    # Database Configuration
    DB_HOST: str
    DB_PORT: int = 5432
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    # Application Settings
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    MAX_RETRIES: int = 3
    TIMEOUT: int = 30

    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        case_sensitive = True
        env_file_encoding = "utf-8"

settings = Settings()
````

## Ruff Configuration

```toml
[tool.ruff]
target-version = "py39"
line-length = 100
fix = true

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors ✔️ 🛠️
    "W",   # pycodestyle warnings ✔️ 🛠️
    "F",   # pyflakes ✔️
    "D",   # pydocstyle ✔️ 🛠️
    "UP",  # pyupgrade ✔️ 🛠️
    "I",   # isort ✔️ 🛠️
]

[tool.ruff.pydocstyle]
convention = "google"
```

````

# Module: error.md
```markdown
# Error Handling and Logging Guide

## Custom Exceptions
```python
from typing import Any, Dict, Optional
from loguru import logger

class BaseError(Exception):
    """Base exception for application errors."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize error with context.

        Args:
            message: Error message
            context: Additional error context
        """
        super().__init__(message)
        self.context = context or {}
        self._log_error()

    def _log_error(self) -> None:
        """Log error with context."""
        logger.bind(**self.context).error(str(self))

class APIError(BaseError):
    """API-related errors."""
    pass

class ValidationError(BaseError):
    """Data validation errors."""
    pass
````

## Logging Configuration

```python
from loguru import logger
import sys

def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure application logging.

    Args:
        log_level: Logging level to use
    """
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="{time} | {level} | {message} | {extra}",
        level=log_level,
        backtrace=True,
        diagnose=True
    )
    logger.add(
        "logs/app.log",
        rotation="500 MB",
        retention="10 days",
        compression="zip"
    )
```

````

# Module: cicd.md
```markdown
# CI/CD Configuration Guide

## GitHub Actions Workflow
```yaml
name: Python CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install Rye
      run: curl -sSf https://rye-up.com/get | bash

    - name: Install dependencies
      run: |
        rye sync
        rye install --dev

    - name: Lint
      run: |
        uv run ruff check .
        uv run ruff format --check .

    - name: Test
      run: |
        uv run pytest tests/ --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
````

````

# Module: project.md
```markdown
# Project Structure Guide

## Standard Layout
````

project_name/
├── .github/
│ └── workflows/
│ └── ci.yml
├── src/
│ └── project_name/
│ ├── **init**.py
│ ├── models/
│ │ ├── **init**.py
│ │ └── core.py
│ ├── services/
│ │ ├── **init**.py
│ │ └── processor.py
│ └── utils/
│ ├── **init**.py
│ └── helpers.py
├── tests/
│ ├── **init**.py
│ ├── conftest.py
│ ├── fixtures/
│ │ └── sample_data.json
│ └── test_processor.py
├── docs/
│ ├── conf.py
│ └── index.rst
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt

````

## File Templates

### __init__.py
```python
"""
Core module for project functionality.

This module provides:
- Main application features
- Core utilities
- Type definitions
"""

__version__ = "0.1.0"
````

### conftest.py

```python
"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path
from typing import Generator

@pytest.fixture
def sample_data() -> Generator[Dict[str, Any], None, None]:
    """Provide sample test data."""
    data = {"key": "value"}
    yield data
```

```

Would you like me to:
1. Add more examples to any module?
2. Create additional modules for specific use cases?
3. Expand the integration examples between modules?

These modules provide a comprehensive foundation but can be expanded based on your specific needs.
```
