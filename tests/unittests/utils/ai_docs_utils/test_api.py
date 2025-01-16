from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from democracy_exe.utils.ai_docs_utils.api import check_prompt_token_size


if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch

    from pytest_mock.plugin import MockerFixture


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
