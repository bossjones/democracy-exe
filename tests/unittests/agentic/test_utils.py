# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportConstantRedefinition=true
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
from __future__ import annotations

import logging
import os
import tempfile

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.embeddings import Embeddings

import pytest

from pytest_mock import MockerFixture

from democracy_exe.agentic._utils import get_or_create_sklearn_index


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture


class TestLogCapture:
    """Custom log capture class for testing."""

    def __init__(self, *args: Any) -> None:
        """Initialize the log capture."""
        self._entries: list[dict[str, Any]] = []

    def msg(self, message: str) -> None:
        """Capture a log message.

        Args:
            message: The message to capture
        """
        self._entries.append({"event": message})

    def __call__(self, *args: Any) -> TestLogCapture:
        """Make the class callable for structlog.

        Returns:
            Self instance for method chaining
        """
        return self


@pytest.fixture(autouse=True)
def setup_test_logging() -> None:
    """Configure structlog for testing.

    This fixture runs automatically before each test.
    """
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.testing.capture_logs,
        ],
        context_class=dict,
        logger_factory=TestLogCapture,
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        cache_logger_on_first_use=False,
    )


@pytest.fixture
def mock_embeddings(mocker: MockerFixture) -> Embeddings:
    """Create a mock embeddings model.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Mock embeddings model
    """
    return mocker.Mock(spec=Embeddings)


@pytest.fixture
def temp_persist_path() -> Generator[Path, None, None]:
    """Create a temporary file path for testing.

    Yields:
        Path: Temporary file path that will be cleaned up after test
    """
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        path = Path(tmp.name)
        yield path
        # Cleanup after test
        if path.exists():
            path.unlink()


def test_create_new_vector_store(mock_embeddings: Embeddings, temp_persist_path: Path) -> None:
    """Test creating a new vector store.

    Args:
        mock_embeddings: Mock embeddings model
        temp_persist_path: Temporary file path
    """
    # Remove temp file if it exists to test creation
    if temp_persist_path.exists():
        temp_persist_path.unlink()

    with structlog.testing.capture_logs() as captured:
        # Create new vector store
        vector_store = get_or_create_sklearn_index(embeddings=mock_embeddings, persist_path=temp_persist_path)

        # Verify vector store was created
        assert isinstance(vector_store, SKLearnVectorStore)
        assert vector_store._persist_path == str(temp_persist_path)

        # Verify log messages
        assert any("Creating new vector store" in log.get("event", "") for log in captured), (
            "Expected 'Creating new vector store' log message not found"
        )


def test_load_existing_vector_store(mock_embeddings: Embeddings, temp_persist_path: Path) -> None:
    """Test loading an existing vector store.

    Args:
        mock_embeddings: Mock embeddings model
        temp_persist_path: Temporary file path
    """
    # First create a vector store
    initial_store = get_or_create_sklearn_index(embeddings=mock_embeddings, persist_path=temp_persist_path)
    initial_store.persist()

    with structlog.testing.capture_logs() as captured:
        # Load existing vector store
        loaded_store = get_or_create_sklearn_index(embeddings=mock_embeddings, persist_path=temp_persist_path)

        # Verify vector store was loaded
        assert isinstance(loaded_store, SKLearnVectorStore)
        assert loaded_store._persist_path == str(temp_persist_path)

        # Verify log messages
        assert any("Loading existing vector store" in log.get("event", "") for log in captured), (
            "Expected 'Loading existing vector store' log message not found"
        )


def test_create_with_temp_directory(mock_embeddings: Embeddings) -> None:
    """Test creating vector store with automatic temp directory.

    Args:
        mock_embeddings: Mock embeddings model
    """
    with structlog.testing.capture_logs() as captured:
        # Create vector store without persist_path
        vector_store = get_or_create_sklearn_index(embeddings=mock_embeddings)

        # Verify temp path was created
        assert vector_store._persist_path.startswith(tempfile.gettempdir())
        assert "sklearn_vector_store_" in vector_store._persist_path
        assert ".parquet" in vector_store._persist_path

        # Verify log messages
        assert any("Created temporary persist path" in log.get("event", "") for log in captured), (
            "Expected 'Created temporary persist path' log message not found"
        )

        # Cleanup
        if os.path.exists(vector_store._persist_path):
            os.remove(vector_store._persist_path)


def test_different_serializer_formats(mock_embeddings: Embeddings, temp_persist_path: Path) -> None:
    """Test vector store creation with different serializer formats.

    Args:
        mock_embeddings: Mock embeddings model
        temp_persist_path: Temporary file path
    """
    serializers = ["json", "bson", "parquet"]

    for serializer in serializers:
        # Create path for this serializer
        path = temp_persist_path.with_suffix(f".{serializer}")

        # Create vector store
        vector_store = get_or_create_sklearn_index(embeddings=mock_embeddings, persist_path=path, serializer=serializer)

        # Verify
        assert isinstance(vector_store, SKLearnVectorStore)
        assert vector_store._persist_path == str(path)
        assert vector_store._serializer == serializer

        # Cleanup
        if path.exists():
            path.unlink()
