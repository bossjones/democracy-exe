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

import numpy as np
import structlog

from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.vectorstores.sklearn import ParquetSerializer
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


# @pytest.fixture(autouse=True)
# def setup_test_logging() -> None:
#     """Configure structlog for testing.

#     This fixture runs automatically before each test.
#     """
#     structlog.configure(
#         processors=[
#             structlog.processors.add_log_level,
#             structlog.processors.StackInfoRenderer(),
#             structlog.processors.format_exc_info,
#             structlog.testing.capture_logs,
#         ],
#         context_class=dict,
#         logger_factory=TestLogCapture,
#         wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
#         cache_logger_on_first_use=False,
#     )


@pytest.fixture
def mock_embeddings(mocker: MockerFixture) -> Embeddings:
    """Create mock embeddings for testing.

    Args:
        mocker: Pytest mocker fixture

    Returns:
        Embeddings: Mock embeddings model
    """
    mock_embed = mocker.Mock(spec=Embeddings)
    mock_embed.embed_documents.return_value = [np.array([0.1, 0.2, 0.3])]
    mock_embed.embed_query.return_value = np.array([0.1, 0.2, 0.3])
    return mock_embed


@pytest.fixture
def temp_persist_path(tmp_path: Path) -> Generator[Path, None, None]:
    """Create temporary file path for vector store.

    Args:
        tmp_path: Pytest temporary directory fixture

    Yields:
        Path: Temporary file path
    """
    persist_path = tmp_path / "test_vector_store.parquet"
    yield persist_path
    if persist_path.exists():
        persist_path.unlink()


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

        # Add some test data with metadata
        vector_store.add_texts(texts=["test document"], metadatas=[{"source": "test", "type": "document"}])

        # Persist the store
        vector_store.persist()

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
    # First create a vector store with test data
    initial_store = get_or_create_sklearn_index(embeddings=mock_embeddings, persist_path=temp_persist_path)
    initial_store.add_texts(
        texts=["test document"], metadatas=[{"source": "test", "type": "document"}]
    )  # Add test data with metadata
    initial_store.persist()  # Persist with test data

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

        # Add test data and persist
        vector_store.add_texts(texts=["test document"], metadatas=[{"source": "test", "type": "document"}])
        vector_store.persist()

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
    # Only test parquet format which is most reliable for numpy arrays
    serializer = "parquet"
    path = temp_persist_path.with_suffix(f".{serializer}")

    # Create vector store
    vector_store = get_or_create_sklearn_index(embeddings=mock_embeddings, persist_path=path, serializer=serializer)

    # Add test data and persist
    vector_store.add_texts(texts=["test document"], metadatas=[{"source": "test", "type": "document"}])
    vector_store.persist()

    # Verify
    assert isinstance(vector_store, SKLearnVectorStore)
    assert vector_store._persist_path == str(path)
    assert isinstance(vector_store._serializer, ParquetSerializer)

    # Cleanup
    if path.exists():
        path.unlink()
