# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
from __future__ import annotations

import json

from typing import TYPE_CHECKING, Any, cast

import httpx

import pytest

from democracy_exe.utils.ai_docs_utils.api import (
    ASYNC_CLIENT,
    BASIC_DOCS_SYSTEM_PROMPT,
    arequest_message,
    check_prompt_token_size,
)


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from anthropic.types import Message as AnthropicMessage
    from respx import MockRouter

    from pytest_mock.plugin import MockerFixture


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


@pytest.mark.asyncio
@pytest.mark.respx
async def test_arequest_message(respx_mock: MockRouter, mocker: MockerFixture) -> None:
    """Test arequest_message function.

    This test verifies that the function correctly sends messages to Anthropic API
    and returns the expected response.

    Args:
        respx_mock: Respx mock router for HTTP mocking
        mocker: Pytest mocker fixture
    """
    # Patch retry timeout
    mocker.patch(
        "anthropic._base_client.BaseClient._calculate_retry_timeout",
        _low_retry_timeout,
    )

    # Mock response data matching Anthropic API structure
    mock_response_data = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Test response"}],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 100},
    }

    # Get the base URL from the async client
    base_url = str(ASYNC_CLIENT.base_url).rstrip("/")

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

    # Test data
    system_prompt = BASIC_DOCS_SYSTEM_PROMPT
    messages = [{"role": "user", "content": "Test message"}]

    # Call the function
    response = await arequest_message(system_prompt, messages)

    # Verify the response structure
    assert response.id.startswith("msg_")
    assert response.role == "assistant"
    assert response.content[0].text == "Test response"
    assert response.model == "claude-3-opus-20240229"

    # Verify the request was made with correct parameters
    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert request.url.path == "/v1/messages"

    # Verify request headers
    assert request.headers["anthropic-version"] == "2023-06-01"

    # Verify request body
    request_body = json.loads(request.read().decode())
    assert request_body["messages"] == messages
    assert request_body["system"] == system_prompt
    assert request_body["model"] == "claude-3-opus-20240229"
    assert request_body["max_tokens"] == 4096

    # Verify no connection leaks
    assert _get_open_connections(ASYNC_CLIENT) == 0
