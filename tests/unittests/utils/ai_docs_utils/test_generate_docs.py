from __future__ import annotations

import os

from pathlib import Path
from typing import TYPE_CHECKING

from anthropic.types import Message as AnthropicMessage

import pytest

from democracy_exe.utils.ai_docs_utils.generate_docs import SIGNATURE, generate_docs_from_local_repo


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
