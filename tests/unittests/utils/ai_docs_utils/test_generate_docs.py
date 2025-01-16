from __future__ import annotations

import os

from pathlib import Path
from typing import TYPE_CHECKING

from anthropic.types import Message as AnthropicMessage

import pytest

from democracy_exe.utils.ai_docs_utils.generate_docs import (
    REFINED_DOCS_FOLLOW_UP_PROMPT,
    SIGNATURE,
    _extend_docs,
    agenerate_docs_from_local_repo,
    generate_docs_from_local_repo,
)
from democracy_exe.utils.file_functions import tilda


if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_response() -> AnthropicMessage:
    """Create a mock Claude API response.

    Returns:
        AnthropicMessage: Mocked response object
    """
    return AnthropicMessage(
        id="msg_123",
        type="message",
        role="assistant",
        content=[{"type": "text", "text": "# Test Repository\n\nThis is a test repository documentation."}],
        model="claude-3-opus-20240229",
        usage={"input_tokens": 100, "output_tokens": 50},
    )


@pytest.fixture
def mock_refined_response() -> AnthropicMessage:
    """Create a mock Claude API response for refined documentation.

    Returns:
        AnthropicMessage: Mocked refined response object
    """
    return AnthropicMessage(
        id="msg_456",
        type="message",
        role="assistant",
        content=[
            {"type": "text", "text": "# Extended Documentation\n\nThis is extended documentation with more details."}
        ],
        model="claude-3-opus-20240229",
        usage={"input_tokens": 150, "output_tokens": 75},
    )


def test_generate_docs_from_local_repo_with_tilde(
    tmp_path: Path,
    mocker: MockerFixture,
    mock_response: AnthropicMessage,
    monkeypatch: MonkeyPatch,
) -> None:
    """Test generating documentation from a local repository with tilde path.

    Args:
        tmp_path: Pytest fixture for temporary directory
        mocker: Pytest mocker fixture
        mock_response: Mock API response fixture
        monkeypatch: Pytest monkeypatch fixture
    """
    # Setup home directory for tilde expansion
    home_dir = tmp_path / "home" / "user"
    test_dir = home_dir / "test_repo"
    os.makedirs(test_dir, exist_ok=True)
    monkeypatch.setenv("HOME", str(home_dir))

    # Create tilde path
    tilde_path = "~/test_repo"
    expanded_path = str(tilda(tilde_path))

    # Mock dependencies
    mock_extract = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.extract_local_directory", return_value="test_code.txt"
    )
    mock_read = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.read_file", return_value="test code content"
    )
    mock_request = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.request_message", return_value=mock_response
    )

    # Run function with tilde path
    generate_docs_from_local_repo(tilde_path)

    # Verify calls with expanded path
    mock_extract.assert_called_once_with(expanded_path)
    mock_read.assert_called_once_with("test_code.txt")
    mock_request.assert_called_once()

    # Verify request message construction
    messages = mock_request.call_args[0][1]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "test code content" in messages[0]["content"]

    # Verify README.md creation in expanded path
    readme_path = os.path.join(expanded_path, "README.md")
    assert os.path.exists(readme_path)

    with open(readme_path, encoding="utf-8") as f:
        content = f.read()
        assert content.startswith(SIGNATURE)
        assert "Test Repository" in content
        assert "This is a test repository documentation." in content


def test_generate_docs_from_local_repo(
    tmp_path: Path,
    mocker: MockerFixture,
    mock_response: AnthropicMessage,
) -> None:
    """Test generating documentation from a local repository.

    Args:
        tmp_path: Pytest fixture for temporary directory
        mocker: Pytest mocker fixture
        mock_response: Mock API response fixture
    """
    # Mock dependencies
    mock_extract = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.extract_local_directory", return_value="test_code.txt"
    )
    mock_read = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.read_file", return_value="test code content"
    )
    mock_request = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.request_message", return_value=mock_response
    )

    # Test directory
    test_dir = str(tmp_path / "test_repo")
    os.makedirs(test_dir, exist_ok=True)

    # Run function
    generate_docs_from_local_repo(test_dir)

    # Verify calls
    mock_extract.assert_called_once_with(test_dir)
    mock_read.assert_called_once_with("test_code.txt")
    mock_request.assert_called_once()

    # Verify request message construction
    messages = mock_request.call_args[0][1]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "test code content" in messages[0]["content"]

    # Verify README.md creation
    readme_path = os.path.join(test_dir, "README.md")
    assert os.path.exists(readme_path)

    with open(readme_path, encoding="utf-8") as f:
        content = f.read()
        assert content.startswith(SIGNATURE)
        assert "Test Repository" in content
        assert "This is a test repository documentation." in content


@pytest.mark.asyncio
async def test_agenerate_docs_from_local_repo(
    tmp_path: Path,
    mocker: MockerFixture,
    mock_response: AnthropicMessage,
) -> None:
    """Test generating documentation asynchronously from a local repository.

    Args:
        tmp_path: Pytest fixture for temporary directory
        mocker: Pytest mocker fixture
        mock_response: Mock API response fixture
    """
    # Mock dependencies
    mock_extract = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.extract_local_directory", return_value="test_code.txt"
    )
    mock_read = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.read_file", return_value="test code content"
    )
    mock_request = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.arequest_message", return_value=mock_response
    )

    # Test directory
    test_dir = str(tmp_path / "test_repo")
    os.makedirs(test_dir, exist_ok=True)

    # Run function
    await agenerate_docs_from_local_repo(test_dir)

    # Verify calls
    mock_extract.assert_called_once_with(test_dir)
    mock_read.assert_called_once_with("test_code.txt")
    mock_request.assert_called_once()

    # Verify request message construction
    messages = mock_request.call_args[0][1]
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "test code content" in messages[0]["content"]

    # Verify README.md creation
    readme_path = os.path.join(test_dir, "README.md")
    assert os.path.exists(readme_path)

    with open(readme_path, encoding="utf-8") as f:
        content = f.read()
        assert content.startswith(SIGNATURE)
        assert "Test Repository" in content
        assert "This is a test repository documentation." in content


@pytest.mark.asyncio
async def test_extend_docs(
    tmp_path: Path,
    mocker: MockerFixture,
    mock_refined_response: AnthropicMessage,
) -> None:
    """Test extending documentation with refined insights.

    Args:
        tmp_path: Pytest fixture for temporary directory
        mocker: Pytest mocker fixture
        mock_refined_response: Mock refined API response fixture
    """
    # Setup
    repo_name = "test-repo"
    initial_message = "Initial documentation content"
    messages = [{"role": "user", "content": "Initial prompt"}]

    # Mock API request
    mock_request = mocker.patch(
        "democracy_exe.utils.ai_docs_utils.generate_docs.arequest_message", return_value=mock_refined_response
    )

    # Change to temp directory for file operations
    os.chdir(tmp_path)

    # Run function
    await _extend_docs(repo_name, messages, initial_message)

    # Verify message list was extended correctly
    assert len(messages) == 3
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == initial_message
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == REFINED_DOCS_FOLLOW_UP_PROMPT

    # Verify API call
    mock_request.assert_called_once()

    # Verify file creation and content
    output_file = f"{repo_name}-further-docs.md"
    assert os.path.exists(output_file)

    with open(output_file, encoding="utf-8") as f:
        content = f.read()
        assert "Extended Documentation" in content
        assert "This is extended documentation with more details." in content
