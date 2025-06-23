# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
from __future__ import annotations

import datetime
import json
import logging
import os
import sys

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anthropic
import httpx
import pytest_structlog
import respx

from httpx import Response

import pytest

from democracy_exe.utils.ai_docs_utils.api import (
    ASYNC_CLIENT,
    BASIC_DOCS_SYSTEM_PROMPT,
    CLIENT,
    _generate_docs,
    arequest_message,
    check_prompt_token_size,
    read_file,
    request_message,
)


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from anthropic.types import Message as AnthropicMessage
    from pytest_structlog import StructuredLogCapture
    from respx import MockRouter

    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def temp_test_file(tmp_path: Path) -> Path:
    """Create a temporary test file.

    Args:
        tmp_path: Pytest fixture providing temporary directory path

    Returns:
        Path: Path to the temporary test file
    """
    test_file = tmp_path / "test.txt"
    test_content = "Test file content\nLine 2\nLine 3"
    test_file.write_text(test_content)
    return test_file


def test_read_file_success(temp_test_file: Path) -> None:
    """Test read_file function with a valid file.

    This test verifies that the function correctly reads a file's contents
    when the file exists.

    Args:
        temp_test_file: Path to temporary test file
    """
    # Read the test file
    content = read_file(str(temp_test_file))

    # Verify the content
    assert content is not None
    assert "Test file content" in content
    assert "Line 2" in content
    assert "Line 3" in content


def test_read_file_not_found() -> None:
    """Test read_file function with a non-existent file.

    This test verifies that the function raises FileNotFoundError
    when the file does not exist.
    """
    # Try to read a non-existent file
    with pytest.raises(FileNotFoundError) as exc_info:
        read_file("nonexistent_file.txt")

    # Verify the error message
    assert "File not found" in str(exc_info.value)
    assert "nonexistent_file.txt" in str(exc_info.value)


def _low_retry_timeout(*_args: Any, **_kwargs: Any) -> float:
    """Return a low retry timeout for testing.

    Returns:
        float: A small timeout value
    """
    return 0.1


def _get_open_connections(client: Any) -> int:
    """Get number of open connections in client's transport pool.

    Args:
        client: The client to check connections for

    Returns:
        int: Number of open connections
    """
    transport = client._client._transport
    assert isinstance(transport, httpx.HTTPTransport) or isinstance(transport, httpx.AsyncHTTPTransport)

    pool = transport._pool
    return len(pool._requests)


def test_check_prompt_token_size(capsys: CaptureFixture[str]) -> None:
    """Test check_prompt_token_size function.

    This test verifies that the function correctly tokenizes input text using GPT-2
    tokenizer and returns the expected token count.

    Args:
        capsys: Pytest fixture to capture stdout/stderr
    """
    # Test with a simple prompt
    prompt = "Hello, world!"
    token_count = check_prompt_token_size(prompt)

    # GPT-2 tokenizes "Hello, world!" into ['Hello', ',', 'world', '!']
    assert token_count == 4

    # Verify debug output
    captured = capsys.readouterr()
    assert "tokenizer:" in captured.out
    assert "tokenizer type:" in captured.out
    assert "tokens:" in captured.out
    assert "tokens type:" in captured.out

    # Test with a longer prompt
    long_prompt = "This is a longer prompt with multiple words and punctuation marks!"
    long_token_count = check_prompt_token_size(long_prompt)

    # Should have more tokens than the simple prompt
    assert long_token_count > token_count


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
@pytest.mark.asyncio
@pytest.mark.respx
async def test_arequest_message(
    respx_mock: respx.Router,
    log: pytest_structlog.StructuredLogCapture,
) -> None:
    """Test arequest_message function.

    This test verifies that:
    1. The function sends requests to Anthropic API correctly
    2. Response is parsed and returned properly
    3. Proper logging occurs for requests and responses
    4. Error cases are handled and logged appropriately

    Args:
        respx_mock: RESPX mock router
        log: Structured log capture fixture
    """
    # Mock successful response
    mock_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }

    route = respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(200, json=mock_response)
    )

    # Test successful request
    system_prompt = "You are a helpful assistant."
    messages = [{"role": "user", "content": "Hello"}]

    response = await arequest_message(system_prompt, messages)

    # Verify request was made correctly
    assert route.called
    assert route.calls.last.request.headers["anthropic-version"] == "2023-06-01"
    assert route.calls.last.request.headers["content-type"] == "application/json"

    request_body = json.loads(route.calls.last.request.content)
    assert request_body["system"] == system_prompt
    assert request_body["messages"] == messages
    assert request_body["model"] == "claude-3-5-sonnet-20241022"

    # Verify response parsing
    assert response.id == "msg_123"
    assert response.type == "message"
    assert response.role == "assistant"
    assert response.content[0].text == "Hello!"
    assert response.model == "claude-3-5-sonnet-20241022"

    # Verify logging
    assert log.has("Sending message to Anthropic API", num_messages=1), (
        "Expected log message for sending request not found"
    )

    assert log.has(
        "Received response from Anthropic API",
        message_id="msg_123",
        model="claude-3-5-sonnet-20241022",
        stop_reason="end_turn",
    ), "Expected log message for received response not found"

    # Test error cases
    # API Error
    route.mock(return_value=Response(500, json={"error": {"message": "Internal server error"}}))
    with pytest.raises(anthropic.APIStatusError) as exc_info:
        await arequest_message(system_prompt, messages)

    assert log.has("Anthropic API request failed", error_type="APIStatusError"), "Expected error log message not found"

    # Rate limit error
    route.mock(return_value=Response(429, json={"error": {"message": "Rate limit exceeded"}}))
    with pytest.raises(anthropic.RateLimitError):
        await arequest_message(system_prompt, messages)

    assert log.has("Anthropic API request failed", error_type="RateLimitError"), (
        "Expected rate limit error log not found"
    )

    # Authentication error
    route.mock(return_value=Response(401, json={"error": {"message": "Invalid API key"}}))
    with pytest.raises(anthropic.AuthenticationError):
        await arequest_message(system_prompt, messages)

    assert log.has("Anthropic API request failed", error_type="AuthenticationError"), (
        "Expected authentication error log not found"
    )


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
@pytest.mark.respx
def test_request_message(
    respx_mock: MockRouter,
    mocker: MockerFixture,
    log: StructuredLogCapture,
) -> None:
    """Test request_message function.

    This test verifies that:
    1. The function sends requests to Anthropic API correctly
    2. Response is parsed and returned properly
    3. Proper logging occurs for requests and responses
    4. Error cases are handled and logged appropriately

    Args:
        respx_mock: Respx mock router for HTTP mocking
        mocker: Pytest mocker fixture
        log: Structured log capture fixture
    """
    # Patch retry timeout
    mocker.patch(
        "anthropic._base_client.BaseClient._calculate_retry_timeout",
        _low_retry_timeout,
    )

    # Mock successful response
    mock_response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Test response"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 50, "output_tokens": 100},
    }

    # Get the base URL from the sync client
    base_url = str(CLIENT.base_url).rstrip("/")

    # Setup mock response with proper headers
    route = respx_mock.post(f"{base_url}/v1/messages").mock(
        return_value=Response(
            200,
            json=mock_response,
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )
    )

    # Test data
    system_prompt = BASIC_DOCS_SYSTEM_PROMPT
    messages = [{"role": "user", "content": "Test message"}]

    # Call the function
    response = request_message(system_prompt, messages)

    # Verify request was made correctly
    assert route.called
    assert route.calls.last.request.headers["anthropic-version"] == "2023-06-01"
    assert route.calls.last.request.headers["content-type"] == "application/json"

    request_body = json.loads(route.calls.last.request.content)
    assert request_body["system"] == system_prompt
    assert request_body["messages"] == messages
    assert request_body["model"] == "claude-3-5-sonnet-20241022"
    assert request_body["max_tokens"] == 4096

    # Verify response parsing
    assert response.id == "msg_123"
    assert response.type == "message"
    assert response.role == "assistant"
    assert response.content[0].text == "Test response"
    assert response.model == "claude-3-5-sonnet-20241022"

    # Verify logging
    assert log.has("Sending message to Anthropic API", num_messages=1), (
        "Expected log message for sending request not found"
    )

    assert log.has(
        "Received response from Anthropic API",
        message_id="msg_123",
        model="claude-3-5-sonnet-20241022",
        stop_reason="end_turn",
    ), "Expected log message for received response not found"

    # Test error cases
    # API Error
    route.mock(return_value=Response(500, json={"error": {"message": "Internal server error"}}))
    with pytest.raises(anthropic.APIStatusError):
        request_message(system_prompt, messages)

    assert log.has("Anthropic API request failed", error_type="APIStatusError"), "Expected error log message not found"

    # Rate limit error
    route.mock(return_value=Response(429, json={"error": {"message": "Rate limit exceeded"}}))
    with pytest.raises(anthropic.RateLimitError):
        request_message(system_prompt, messages)

    assert log.has("Anthropic API request failed", error_type="RateLimitError"), (
        "Expected rate limit error log not found"
    )

    # Authentication error
    route.mock(return_value=Response(401, json={"error": {"message": "Invalid API key"}}))
    with pytest.raises(anthropic.AuthenticationError):
        request_message(system_prompt, messages)

    assert log.has("Anthropic API request failed", error_type="AuthenticationError"), (
        "Expected authentication error log not found"
    )


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
@pytest.mark.respx
def test_generate_docs(
    respx_mock: MockRouter,
    mocker: MockerFixture,
    temp_test_file: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test _generate_docs function.

    This test verifies that the function correctly generates documentation
    from a file and handles user input.

    Args:
        respx_mock: Respx mock router for HTTP mocking
        mocker: Pytest mocker fixture
        temp_test_file: Temporary test file
        capsys: Fixture to capture stdout/stderr
        monkeypatch: Pytest monkeypatch fixture
    """
    # Create a test file with _code in the name
    test_file = temp_test_file.parent / f"{temp_test_file.stem}_code{temp_test_file.suffix}"
    test_file.write_text(temp_test_file.read_text())

    # Mock token size to return a small number
    def mock_token_size(prompt: str) -> int:
        print("Input token size is: 10")
        return 10

    mocker.patch(
        "democracy_exe.utils.ai_docs_utils.api.check_prompt_token_size",
        side_effect=mock_token_size,
    )

    # Mock user input to proceed
    monkeypatch.setattr("builtins.input", lambda _: "Y")

    # Mock response data matching Anthropic API structure
    mock_response_data = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "# Generated Documentation\n\nTest content"}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 100},
    }

    # Get the base URL from the sync client
    base_url = str(CLIENT.base_url).rstrip("/")

    # Setup mock response with proper headers
    respx_mock.post(f"{base_url}/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json=mock_response_data,
            headers={
                "Content-Type": "application/json",
                "X-Anthropic-Version": "2023-06-01",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "messages-2024-02-29",
            },
        )
    )

    # Call the function
    _generate_docs(str(test_file))

    # Verify output file was created
    output_file = test_file.parent / f"{test_file.stem.replace('_code', '')}{test_file.suffix}-docs.md"
    assert output_file.exists(), f"Output file not found at {output_file}"

    # Verify file content
    content = output_file.read_text()
    assert "# Generated Documentation" in content
    assert "Test content" in content

    # Verify console output
    captured = capsys.readouterr()
    assert "Input token size is: 10" in captured.out

    # Clean up the generated files
    test_file.unlink()
    output_file.unlink()


@pytest.mark.skip_until(
    deadline=datetime.datetime(2025, 1, 25),
    strict=True,
    msg="Need to find a good url to test this with, will do later",
)
def test_generate_docs_user_abort(
    mocker: MockerFixture,
    temp_test_file: Path,
    capsys: CaptureFixture[str],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test _generate_docs function when user aborts.

    This test verifies that the function correctly handles user abort
    and exits gracefully.

    Args:
        mocker: Pytest mocker fixture
        temp_test_file: Temporary test file
        capsys: Fixture to capture stdout/stderr
        monkeypatch: Pytest monkeypatch fixture
    """
    # Create a test file with _code in the name
    test_file = temp_test_file.parent / f"{temp_test_file.stem}_code{temp_test_file.suffix}"
    test_file.write_text(temp_test_file.read_text())

    # Mock token size to return a small number
    def mock_token_size(prompt: str) -> int:
        print("Input token size is: 10")
        return 10

    mocker.patch(
        "democracy_exe.utils.ai_docs_utils.api.check_prompt_token_size",
        side_effect=mock_token_size,
    )

    # Mock user input to abort
    monkeypatch.setattr("builtins.input", lambda _: "N")

    # Mock sys.exit to prevent actual exit
    mock_exit = mocker.patch("sys.exit")

    # Call the function
    _generate_docs(str(test_file))

    # Verify sys.exit was called with code 1
    mock_exit.assert_called_once_with(1)

    # Verify console output
    captured = capsys.readouterr()
    assert "Input token size is: 10" in captured.out
    assert "Exiting" in captured.out

    # Clean up test file
    test_file.unlink()


@pytest.fixture(autouse=True)
def setup_structlog() -> None:
    """Configure structlog for testing.

    This fixture ensures proper processor ordering for test capture.
    """
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            pytest_structlog.StructuredLogCapture,
            structlog.dev.ConsoleRenderer(),  # ConsoleRenderer must be last
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
