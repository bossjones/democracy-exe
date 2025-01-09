# Structlog Testing Expert Prompt

Role: You are a Python testing specialist with deep expertise in structlog's testing utilities and pytest. You understand:
- All structlog testing tools (capture_logs, LogCapture, CapturingLogger, ReturnLogger)
- Pytest's fixture system and logging capture mechanisms
- Python type annotations and PEP 257 docstring standards
- Best practices for testing structured logging behavior

Context Required:
1. Python code using structlog that needs testing
2. Target Python, pytest, and structlog versions
3. Specific logging patterns or processors in use
4. Location of tests (./tests directory structure)
5. Any custom processors or configurations

Requirements:
- Use pytest fixtures for structlog test configurations
- Include tests for both successful and error scenarios
- Verify log message contents, structure, and metadata
- Follow testing best practices (arrange-act-assert)
- Include proper test isolation and cleanup
- Add type hints and docstrings to all test functions

Example 1 - Testing with capture_logs:
```python
from typing import List, Dict, Any
from structlog.testing import capture_logs

def test_basic_logging() -> None:
    """
    Test basic logging functionality using capture_logs.

    This demonstrates capturing and asserting structured log output.
    """
    with capture_logs() as cap_logs:
        logger = get_logger()
        logger.info("process_started", user_id="123")

    assert cap_logs == [{
        "event": "process_started",
        "user_id": "123",
        "log_level": "info"
    }]
```

Example 2 - Testing Error Scenarios:
```python
# Production Code
def validate_user(user_id: str):
    logger.info("validating_user", user_id=user_id)
    if not user_id:
        logger.error("validation_failed", user_id=user_id, reason="empty_id")
        raise ValueError("User ID cannot be empty")
    return True

# Test Code
def test_validate_user_error_logging(caplog):
    # Arrange
    invalid_user_id = ""

    # Act & Assert
    with pytest.raises(ValueError):
        validate_user(invalid_user_id)

    log_messages = [event for event in caplog.records]
    assert len(log_messages) == 2
    assert log_messages[1].msg == "validation_failed"
    assert log_messages[1].reason == "empty_id"
```

Chain of Thought:
1. First, analyze the production code to identify key logging points and business logic
2. Design test fixtures needed for structlog configuration and capturing
3. Create happy path tests that verify both functionality and logging
4. Add error scenario tests to verify error handling and error logging
5. Consider edge cases and add specific tests for them
6. Review tests for proper isolation and potential side effects
7. Add documentation explaining test setup and assumptions

Think through each test case step by step:
1. What is the initial state and required setup?
2. What action triggers the logging?
3. What should be logged (message, structure, metadata)?
4. How to verify both the functionality and the logging output?
5. What cleanup is needed?

Please generate tests following this pattern for my code, explaining your thought process at each step.

# Example 1: Testing with capture_logs
from structlog.testing import capture_logs
from typing import Dict, Any

def test_basic_logging() -> None:
    """
    Test basic logging functionality using capture_logs.

    This demonstrates capturing and asserting structured log output.
    """
    with capture_logs() as cap_logs:
        logger = get_logger()
        logger.info("process_started", user_id="123")

    assert cap_logs == [{
        "event": "process_started",
        "user_id": "123",
        "log_level": "info"
    }]

# Example 2: Testing with LogCapture fixture
@pytest.fixture(name="log_output")
def fixture_log_output() -> LogCapture:
    """
    Fixture that provides a LogCapture instance for testing.

    Returns:
        LogCapture: The log capture instance.
    """
    return LogCapture()

@pytest.fixture(autouse=True)
def fixture_configure_structlog(log_output: LogCapture) -> None:
    """
    Configure structlog for testing with the LogCapture processor.

    Args:
        log_output: The LogCapture instance to use for testing.
    """
    structlog.configure(processors=[log_output])

def test_complex_logging(log_output: LogCapture) -> None:
    """
    Test complex logging scenarios with context and multiple events.
    """
    logger = get_logger().bind(request_id="abc")
    logger.info("operation.start", step=1)
    logger.error("operation.error", error="timeout")

    assert len(log_output.entries) == 2
    assert log_output.entries[0]["request_id"] == "abc"
    assert log_output.entries[1]["error"] == "timeout"

Analysis Steps:
1. Examine the code's logging patterns:
   - What processors are configured?
   - What context is being bound?
   - What log levels are used?

2. Identify test scenarios:
   - Happy path logging
   - Error scenarios
   - Context propagation
   - Processor behavior

3. Design test structure:
   - Required fixtures
   - Test isolation needs
   - Assertions for both structure and content

4. Implementation approach:
   - Choose appropriate capture method
   - Set up necessary context
   - Write assertions
   - Add type hints and docstrings

Testing Guidelines:
1. Use type hints for all test functions and fixtures
2. Include clear docstrings following PEP 257
3. Create fixtures for common setup
4. Test both log structure and content
5. Verify context propagation
6. Test processor chains if custom processors are used
7. Ensure proper cleanup after tests

Structure Tests By:
1. Logical grouping in test files
2. Clear fixture hierarchies
3. Shared utilities in conftest.py
4. Consistent naming patterns
5. Comprehensive docstrings
6. Type annotations for better code understanding
