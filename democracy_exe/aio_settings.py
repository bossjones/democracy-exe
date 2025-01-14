# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportConstantRedefinition=true

"""Settings for the Discord bot and AI components.

This module manages configuration settings with comprehensive validation,
error handling, and security checks. It uses Pydantic for validation
and supports environment variable configuration.

Implementation Details:
    - Configuration Management: Uses Pydantic with custom validators
    - Error Handling: Hierarchical error classes for different scenarios
    - Security: Comprehensive security validation for sensitive fields
    - Model Configuration: Token limits and pricing for different models
    - Resource Management: Memory, task, and buffer size limits
    - Database Integration: Redis and PostgreSQL configuration
    - Environment Variables: Prefix DEMOCRACY_EXE_CONFIG_

Missing or Needs Improvement:
    - Comprehensive test coverage for all validators
    - Migration path for configuration updates
    - Default value documentation
    - Configuration versioning system
    - Environment variable documentation
    - Configuration backup and restore
    - Secret rotation mechanisms
    - Configuration audit logging
    - Dynamic reconfiguration support
    - Configuration schema versioning

Security Notes:
    - Sensitive values use SecretStr
    - Password validation rules enforced
    - URL validation for database connections
    - Token validation for API keys
    - Rate limit configuration validation

Usage:
    Load settings from environment:
    ```python
    from democracy_exe.aio_settings import aiosettings

    # Access settings
    model_name = aiosettings.llm_model_name
    max_tokens = aiosettings.llm_max_tokens

    # Override from environment:
    # export DEMOCRACY_EXE_CONFIG_LLM_MODEL_NAME=gpt-4-0613
    ```
"""
from __future__ import annotations

import enum
import os
import pathlib

from datetime import timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Union, cast

import rich

from pydantic import Field, Json, PostgresDsn, RedisDsn, SecretStr, field_serializer, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.console import Console
from typing_extensions import Self, TypedDict
from yarl import URL


def democracy_user_agent() -> str:
    """Get a common user agent"""
    return "democracy-exe/0.0.1"


# Get rid of warning
# USER_AGENT environment variable not set, consider setting it to identify your requests.
# os.environ["USER_AGENT"] = democracy_user_agent()

TIMEZONE = timezone(timedelta(hours=-5), name='America/New_York')

_TOKENS_PER_TILE = 170
_TILE_SIZE = 512



# Older model configurations
_OLDER_MODEL_CONFIG: dict[str, dict[str, int | float]] = {
    "gpt-4-0613": {
        "max_tokens": 8192,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00003,
        "completion_cost_per_token": 0.00006,
    },
    "gpt-4-32k-0314": {
        "max_tokens": 32768,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00006,
        "completion_cost_per_token": 0.00012,
    },
    "gpt-4-32k-0613": {
        "max_tokens": 32768,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00006,
        "completion_cost_per_token": 0.00012,
    },
    "gpt-3.5-turbo-0301": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-0613": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-16k-0613": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000003,
        "completion_cost_per_token": 0.000004,
    },
    "gpt-3.5-turbo-instruct": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    }
}

# SOURCE: https://github.com/JuliusHenke/autopentest/blob/ca822f723a356ec974d2dff332c2d92389a4c5e3/src/text_embeddings.py#L19
# https://platform.openai.com/docs/guides/embeddings/embedding-models
_OLDER_EMBEDDING_CONFIG: dict[str, dict[str, int | float]] = {
    "text-embedding-ada-002": {
        "max_tokens": 8191,
        "prompt_cost_per_token": 0.0000001,
        "completion_cost_per_token": 0.0
    }
}

# Model configurations
_NEWER_MODEL_CONFIG: dict[str, dict[str, int | float]] = {
    "claude-3-5-sonnet-20240620": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "claude-3-opus-20240229": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "claude-3-sonnet-20240229": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "claude-3-haiku-20240307": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "gpt-4o-2024-08-06": {
        "max_tokens": 128000,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.00001,
    },
    "gpt-4o-mini-2024-07-18": {
        "max_tokens": 900,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.000000150,
        "completion_cost_per_token": 0.00000060,
    },
    "gpt-4o-2024-05-13": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000005,
        "completion_cost_per_token": 0.000015,
    },
    "gpt-4-turbo-2024-04-09": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-4-0125-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-4-1106-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-4-vision-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-3.5-turbo-0125": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0000005,
        "completion_cost_per_token": 0.0000015,
    },
    "gpt-3.5-turbo-1106": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000001,
        "completion_cost_per_token": 0.000002,
    },
    "gemma-7b-it": {
        "max_tokens": 8192,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0,  # Open source model
        "completion_cost_per_token": 0.0,
    },
    "gemma2-9b-it": {
        "max_tokens": 8192,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0,  # Open source model
        "completion_cost_per_token": 0.0,
    },
    "llama3-70b-8192": {
        "max_tokens": 8192,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0,  # Open source model
        "completion_cost_per_token": 0.0,
    },
    "llama3-8b-8192": {
        "max_tokens": 8192,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0,  # Open source model
        "completion_cost_per_token": 0.0,
    },
    "mixtral-8x7b-32768": {
        "max_tokens": 32768,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.0,  # Open source model
        "completion_cost_per_token": 0.0,
    }
}

# SOURCE: https://github.com/JuliusHenke/autopentest/blob/ca822f723a356ec974d2dff332c2d92389a4c5e3/src/text_embeddings.py#L19
# https://platform.openai.com/docs/guides/embeddings/embedding-models
_NEWER_EMBEDDING_CONFIG: dict[str, dict[str, int | float]] = {
    "text-embedding-3-small": {
        "max_tokens": 8191,
        "prompt_cost_per_token": 0.00000002,
        "completion_cost_per_token": 0.0
    },
    "text-embedding-3-large": {
        "max_tokens": 8191,
        "prompt_cost_per_token": 0.00000013,
        "completion_cost_per_token": 0.0
    }
}

# Model aliases for easier reference
MODEL_POINT: dict[str, str] = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "gpt-4": "gpt-4-0613",
    "gpt-4-32k": "gpt-4-32k-0613",
    "gpt-4-vision": "gpt-4-vision-preview",
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k-0613",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20240620",

    # "gpt4": "gpt-4-0613",
    # "gpt4-32k": "gpt-4-32k-0613",
    # "gpt4-turbo": "gpt-4-turbo-2024-04-09",
    # "gpt4-vision": "gpt-4-vision-preview",
    # "gpt35": "gpt-3.5-turbo-0125",
    # "gpt35-16k": "gpt-3.5-turbo-16k-0613",
    # "claude3-opus": "claude-3-opus-20240229",
    # "claude3-sonnet": "claude-3-sonnet-20240229",
    # "claude3-haiku": "claude-3-haiku-20240307",
    # "gpt4o": "gpt-4o-2024-08-06",
    # "gpt4o-mini": "gpt-4o-mini-2024-07-18",
    # "gemma7b": "gemma-7b-it",
    # "gemma9b": "gemma2-9b-it",
    # "llama70b": "llama3-70b-8192",
    # "llama8b": "llama3-8b-8192",
    # "mixtral": "mixtral-8x7b-32768"
}

# Combine configurations
MODEL_CONFIG: dict[str, dict[str, int | float]] = {**_OLDER_MODEL_CONFIG, **_NEWER_MODEL_CONFIG}

# Combine embedding configurations
EMBEDDING_CONFIG: dict[str, dict[str, int | float]] = {**_OLDER_EMBEDDING_CONFIG, **_NEWER_EMBEDDING_CONFIG}

# Add aliased configurations
_MODEL_POINT_CONFIG = {
    alias: MODEL_CONFIG[target]
    for alias, target in MODEL_POINT.items()
}
MODEL_CONFIG.update(_MODEL_POINT_CONFIG)

# produces a list of all models and embeddings available
MODEL_ZOO = set(MODEL_CONFIG.keys()) | set(EMBEDDING_CONFIG.keys())

# Embedding model dimensions for different models
EMBEDDING_MODEL_DIMENSIONS_DATA: dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 1024
}

class SettingsError(Exception):
    """Base exception for all settings-related errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the settings error.

        Args:
            message: The error message
            context: Additional context about the error
        """
        self.context = context or {}
        super().__init__(message)


class ValidationError(SettingsError):
    """Raised when settings validation fails."""

    def __init__(self, message: str, field_name: str, invalid_value: Any, context: dict[str, Any] | None = None) -> None:
        """Initialize the validation error.

        Args:
            message: The error message
            field_name: Name of the field that failed validation
            invalid_value: The value that failed validation
            context: Additional context about the error
        """
        self.field_name = field_name
        self.invalid_value = invalid_value
        super().__init__(
            f"{message} (field: {field_name}, value: {invalid_value})",
            context
        )


class ModelConfigError(SettingsError):
    """Raised when there are issues with model configuration."""

    def __init__(self, message: str, model_name: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the model configuration error.

        Args:
            message: The error message
            model_name: Name of the model with configuration issues
            context: Additional context about the error
        """
        self.model_name = model_name
        super().__init__(
            f"{message} (model: {model_name})",
            context
        )


class SecurityError(SettingsError):
    """Raised when there are security-related issues with settings."""

    def __init__(self, message: str, setting_name: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the security error.

        Args:
            message: The error message
            setting_name: Name of the setting with security issues
            context: Additional context about the error
        """
        self.setting_name = setting_name
        super().__init__(
            f"Security error with setting '{setting_name}': {message}",
            context
        )


class ConfigurationError(SettingsError):
    """Raised when there are issues with the overall configuration."""

    def __init__(self, message: str, config_section: str, context: dict[str, Any] | None = None) -> None:
        """Initialize the configuration error.

        Args:
            message: The error message
            config_section: Section of configuration with issues
            context: Additional context about the error
        """
        self.config_section = config_section
        super().__init__(
            f"Configuration error in section '{config_section}': {message}",
            context
        )



def normalize_settings_path(file_path: str) -> str:
    """Normalize file paths with tilde expansion.

    Converts paths starting with ~ to absolute paths using the user's home directory.

    Args:
        file_path: Path to normalize, may start with ~

    Returns:
        str: Normalized absolute path
    """
    return str(pathlib.Path(file_path).expanduser()) if file_path.startswith("~") else file_path

class LogLevel(str, enum.Enum):
    """Possible log levels for application logging.

    Standard Python logging levels used throughout the application.
    """
    NOTSET = "NOTSET"  # 0
    DEBUG = "DEBUG"    # 10
    INFO = "INFO"      # 20
    WARNING = "WARNING"  # 30
    ERROR = "ERROR"    # 40
    FATAL = "FATAL"    # 50

class AioSettings(BaseSettings):
    """Settings for the Discord bot and AI components.

    This class manages all configuration settings for the application,
    supporting environment variable overrides and validation. Settings
    are grouped by functionality (monitor, development, bot, etc.) and
    include comprehensive validation and type checking.

    Key Features:
        - Environment variable configuration with prefix DEMOCRACY_EXE_CONFIG_
        - Automatic model token limit updates based on model selection
        - Comprehensive retry configuration for LLM calls
        - Integrated Redis and PostgreSQL support
        - QA and summarization configuration

    Example:
        ```python
        settings = AioSettings()

        # Access settings
        model_name = settings.llm_model_name
        max_tokens = settings.llm_max_tokens

        # Use Redis URL
        redis_url = settings.redis_url

        # Override from environment:
        # export DEMOCRACY_EXE_CONFIG_LLM_MODEL_NAME=gpt-4-0613
        ```
    """
    model_config = SettingsConfigDict(
        env_prefix="DEMOCRACY_EXE_CONFIG_",
        env_file=(".env", ".envrc"),
        env_file_encoding="utf-8",
        extra="allow",
        arbitrary_types_allowed=True,
        json_schema_extra={
            "properties": {
                "llm_retriever_type": {
                    "type": "string",
                    "default": "vector_store",
                    "description": "Type of retriever to use",
                }
            }
        },
    )

    # Monitor settings
    monitor_host: str = Field(
        default="localhost",
        description="Host for monitoring server"
    )
    monitor_port: int = Field(
        default=50102,
        description="Port for monitoring server"
    )

    # Debug settings
    debug_langchain: bool | None = Field(
        default=False,
        description="Enable LangChain debug logging"
    )

    # Audit settings
    audit_log_send_channel: str = Field(
        default="",
        description="Channel ID for audit log messages"
    )

    # OpenCommit settings
    oco_openai_api_key: SecretStr = Field(
        env="OCO_OPENAI_API_KEY",
        default=SecretStr(""),
        description="OpenAI API key for OpenCommit"
    )
    oco_tokens_max_input: int = Field(
        env="OCO_TOKENS_MAX_INPUT",
        default=4096,
        description="Maximum input tokens for OpenCommit"
    )
    oco_tokens_max_output: int = Field(
        env="OCO_TOKENS_MAX_OUTPUT",
        default=500,
        description="Maximum output tokens for OpenCommit"
    )
    oco_model: str = Field(
        env="OCO_MODEL",
        default="gpt-4o-mini-2024-07-18",
        description="Model to use for OpenCommit"
    )
    oco_language: str = Field(
        env="OCO_LANGUAGE",
        default="en",
        description="Language for OpenCommit messages"
    )
    oco_prompt_module: str = Field(
        env="OCO_PROMPT_MODULE",
        default="conventional-commit",
        description="Prompt module for OpenCommit"
    )
    oco_ai_provider: str = Field(
        env="OCO_AI_PROVIDER",
        default="openai",
        description="AI provider for OpenCommit"
    )

    # OpenAI specific settings
    openai_embeddings_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI model for embeddings"
    )

    # LLM retriever settings
    llm_retriever_type: str = Field(
        default="vector_store",
        description="Type of retriever to use"
    )

    # Development settings
    dev_mode: bool = Field(
        default=False,
        description="Enable development mode for additional debugging and error handling"
    )
    better_exceptions: int = Field(
        default=1,
        description="Enable better exception formatting"
    )
    pythonasynciodebug: bool = Field(
        env="PYTHONASYNCIODEBUG", description="enable or disable asyncio debugging", default=0
    )

    # Bot settings
    prefix: str = Field(
        default="?",
        description="Command prefix for the Discord bot"
    )
    discord_command_prefix: str = "?"
    discord_client_id: int | str = Field(
        default=0,
        description="Discord application client ID"
    )
    discord_client_secret: SecretStr = Field(
        default=SecretStr(""),
        description="Discord application client secret"
    )
    discord_token: SecretStr = Field(
        default=SecretStr(""),
        description="Discord bot token"
    )
    discord_admin_user_id: int = Field(
        default=3282,
        description="Discord user ID of the bot administrator"
    )
    discord_general_channel: int = Field(
        default=908894727779258390,
        description="Discord channel ID for general messages"
    )
    discord_admin_user_invited: bool = Field(
        default=False,
        description="Whether the admin user has been invited to the bot"
    )
    enable_ai: bool = Field(
        default=True,
        description="Enable AI features"
    )
    rate_limit_rate: int = Field(
        default=10,
        description="Number of commands allowed per time period"
    )
    rate_limit_per: float = Field(
        default=12.0,
        description="Time period in seconds for rate limiting"
    )
    spam_ban_threshold: int = Field(
        default=5,
        description="Number of violations before spam ban"
    )
    max_queue_size: int = Field(
        default=1000,
        description="Maximum size of the command queue"
    )
    num_workers: int = Field(
        default=3,
        description="Number of worker threads"
    )

    # Resource management settings
    max_memory_mb: int = Field(
        default=512,
        description="Maximum memory usage in megabytes"
    )
    max_tasks: int = Field(
        default=100,
        description="Maximum number of concurrent tasks"
    )
    max_response_size_mb: int = Field(
        default=8,
        description="Maximum response size in megabytes"
    )
    max_buffer_size_kb: int = Field(
        default=64,
        description="Maximum buffer size in kilobytes"
    )
    task_timeout_seconds: int = Field(
        default=300,
        description="Task timeout in seconds"
    )

    # LLM settings
    llm_model_name: str = Field(
        default="gpt-4o-mini",
        description="The chat model to use"
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Temperature for LLM sampling"
    )
    llm_max_tokens: int = Field(
        default=1024,
        description="Maximum tokens for LLM response"
    )
    llm_top_p: float = Field(
        default=1.0,
        description="Top p for nucleus sampling"
    )
    llm_frequency_penalty: float = Field(
        default=0.0,
        description="Frequency penalty for token generation"
    )
    llm_presence_penalty: float = Field(
        default=0.0,
        description="Presence penalty for token generation"
    )
    llm_stop: list[str] = Field(
        default_factory=list,
        description="Stop sequences for LLM"
    )

    # LLM retry settings
    llm_max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for LLM calls"
    )
    llm_retry_delay: int = Field(
        default=1,
        description="Initial retry delay in seconds"
    )
    llm_retry_max_delay: int = Field(
        default=60,
        description="Maximum retry delay in seconds"
    )
    llm_retry_multiplier: float = Field(
        default=2.0,
        description="Multiplier for retry delay"
    )
    llm_retry_codes: list[int] = Field(
        default_factory=list,
        description="HTTP status codes to retry on"
    )
    llm_retry_methods: list[str] = Field(
        default_factory=list,
        description="HTTP methods to retry"
    )
    llm_retry_backoff: bool = Field(
        default=True,
        description="Use exponential backoff for retries"
    )
    llm_retry_jitter: bool = Field(
        default=True,
        description="Add jitter to retry delays"
    )

    # Redis settings
    redis_host: str = Field(
        default="localhost",
        description="Redis server host"
    )
    redis_port: int = Field(
        default=8600,
        description="Redis server port"
    )
    redis_user: str | None = Field(
        default=None,
        description="Redis username"
    )
    redis_pass: SecretStr | None = Field(
        default=None,
        description="Redis password"
    )
    redis_base: int | None = Field(
        default=None,
        description="Redis database number"
    )
    enable_redis: bool = Field(
        default=True,
        description="Enable Redis integration"
    )
    redis_url: URL | str | None = None

    # PostgreSQL settings
    postgres_host: str = Field(
        default="localhost",
        description="PostgreSQL server host"
    )
    postgres_port: int = Field(
        default=8432,
        description="PostgreSQL server port"
    )
    postgres_user: str = Field(
        default="langchain",
        description="PostgreSQL username"
    )
    postgres_password: str = Field(
        default="langchain",
        description="PostgreSQL password"
    )
    postgres_driver: str = Field(
        default="psycopg",
        description="PostgreSQL driver"
    )
    postgres_database: str = Field(
        default="langchain",
        description="PostgreSQL database name"
    )
    postgres_collection_name: str = Field(
        default="langchain",
        description="PostgreSQL collection name"
    )
    enable_postgres: bool = Field(
        default=True,
        description="Enable PostgreSQL integration"
    )

    # QA settings
    qa_no_chat_history: bool = Field(
        default=False,
        description="Disable chat history for QA"
    )
    qa_followup_sim_threshold: float = Field(
        default=0.735,
        description="Similarity threshold for followup questions"
    )
    qa_retriever: dict[str, Any] = Field(
        default_factory=dict,
        description="QA retriever configuration"
    )

    # Global settings
    globals_try_patchmatch: bool = Field(
        default=True,
        description="Try patch matching in global scope"
    )
    globals_always_use_cpu: bool = Field(
        default=False,
        description="Force CPU usage for operations"
    )
    globals_internet_available: bool = Field(
        default=True,
        description="Whether internet access is available"
    )
    globals_full_precision: bool = Field(
        default=False,
        description="Use full precision for calculations"
    )
    globals_ckpt_convert: bool = Field(
        default=False,
        description="Convert checkpoints"
    )
    globals_log_tokenization: bool = Field(
        default=False,
        description="Log tokenization details"
    )

    # Additional debug settings
    debug_langchain: bool = Field(
        default=True,
        description="Enable LangChain debug logging"
    )
    python_debug: bool = Field(
        default=False,
        description="Enable Python debug mode"
    )
    pythondevmode: bool = Field(
        env="PYTHONDEVMODE",
        description="The Python Development Mode introduces additional runtime checks that are too expensive to be enabled by default. It should not be more verbose than the default if the code is correct; new warnings are only emitted when an issue is detected.",
        default=0,
    )
    local_test_debug: bool = Field(
        default=False,
        description="Enable local test debugging"
    )
    local_test_enable_evals: bool = Field(
        default=False,
        description="Enable evaluation tests locally"
    )
    http_client_debug_enabled: bool = Field(
        default=False,
        description="Enable HTTP client debugging"
    )

    # LLM streaming settings
    llm_streaming: bool = Field(
        default=False,
        description="Whether to use streaming responses from LLM"
    )

    # Feature flags
    rag_answer_accuracy_feature_flag: bool = Field(
        default=False,
        description="Enable RAG answer accuracy evaluation"
    )
    rag_answer_v_reference_feature_flag: bool = Field(
        default=False,
        description="Enable RAG answer vs reference comparison"
    )
    compare_models_feature_flag: bool = Field(
        default=False,
        description="Enable model comparison feature"
    )
    helpfulness_evaluation_feature_flag: bool = Field(
        default=False,
        description="Enable helpfulness evaluation"
    )
    document_relevance_feature_flag: bool = Field(
        default=False,
        description="Enable document relevance evaluation"
    )
    helpfulness_feature_flag: bool = Field(
        default=False,
        description="Enable helpfulness evaluation"
    )
    helpfulness_testing_feature_flag: bool = Field(
        default=False,
        description="Enable helpfulness testing feature"
    )

    # Discord settings
    discord_server_id: int = Field(
        default=1234567890,
        description="Discord server ID"
    )

    # Model configurations
    MODEL_POINT: dict[str, str] = MODEL_POINT
    MODEL_CONFIG: dict[str, dict[str, int | float]] = MODEL_CONFIG
    EMBEDDING_CONFIG: dict[str, dict[str, int | float]] = EMBEDDING_CONFIG
    EMBEDDING_MODEL_DIMENSIONS_DATA: dict[str, int] = EMBEDDING_MODEL_DIMENSIONS_DATA
    MODEL_ZOO: set[str] = MODEL_ZOO

    # LLM Provider settings
    llm_provider: str = Field(
        default="openai",
        description="The LLM provider to use (openai, anthropic, etc.)"
    )
    llm_document_loader_type: str = Field(
        default="pymupdf",
        description="The document loader type to use"
    )
    llm_embedding_model_type: str = Field(
        default="text-embedding-3-large",
        description="The embedding model type to use"
    )

    # Vector store settings
    enable_chroma: bool = Field(
        default=False,
        description="Enable Chroma vector store"
    )

    # Feature flags
    rag_answer_hallucination_feature_flag: bool = Field(
        default=False,
        description="Enable RAG answer hallucination detection"
    )

    enable_sentry: bool = Field(
        default=False,
        description="Enable Sentry error tracking"
    )

    rag_doc_relevance_feature_flag: bool = Field(
        default=False,
        description="Enable RAG document relevance evaluation"
    )

    llm_vectorstore_type: str = Field(
        default="pgvector",
        description="The vector store type to use"
    )

    experimental_redis_memory: bool = Field(
        default=False,
        description="Enable experimental Redis memory features"
    )

    rag_doc_relevance_and_hallucination_feature_flag: bool = Field(
        default=False,
        description="Enable RAG document relevance and hallucination detection"
    )

    # OpenAI settings
    openai_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="OpenAI API key"
    )

    # Feature flags
    rag_string_embedding_distance_metrics_feature_flag: bool = Field(
        default=False,
        description="Enable RAG string embedding distance metrics"
    )

    pinecone_api_key: SecretStr = Field(
        env="PINECONE_API_KEY",
        description="pinecone api key",
        default=SecretStr("")
    )
    pinecone_env: str = Field(
        env="PINECONE_ENV",
        description="pinecone env",
        default="local"
    )
    pinecone_index: str = Field(
        env="PINECONE_INDEX",
        description="pinecone index",
        default=""
    )
    pinecone_namespace: str = Field(
        env="PINECONE_NAMESPACE",
        description="pinecone namespace",
        default="ns1"
    )
    pinecone_index_name: str = Field(
        env="PINECONE_INDEX_NAME",
        description="pinecone index name",
        default="democracy-exe"
    )
    pinecone_url: str = Field(
        env="PINECONE_URL",
        description="pinecone url",
        default="https://democracy-exe-dxt6ijd.svc.aped-4627-b74a.pinecone.io"
    )

    chatbot_type: Literal["terminal", "discord"] = Field(
        env="CHATBOT_TYPE",
        description="chatbot type",
        default="terminal"
    )

    cohere_api_key: SecretStr = Field(
        env="COHERE_API_KEY",
        description="cohere api key",
        default=SecretStr("")
    )
    anthropic_api_key: SecretStr = Field(
        env="ANTHROPIC_API_KEY",
        description="claude api key",
        default=SecretStr("")
    )

    chat_history_buffer: int = Field(
        default=10,
        description="Number of messages to keep in chat history"
    )

    chat_model: str = Field(
        default="gpt-4o-mini",
        description="The chat model to use"
    )

    vision_model: str = Field(env="VISION_MODEL", description="vision model", default="gpt-4o")

    eval_max_concurrency: int = Field(
        default=4,
        description="Maximum number of concurrent evaluation tasks"
    )

    groq_api_key: SecretStr = Field(
        env="GROQ_API_KEY",
        description="groq api key",
        default=SecretStr("")
    )

    langchain_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for LangChain"
    )

    langchain_debug_logs: bool = Field(
        env="LANGCHAIN_DEBUG_LOGS", description="enable or disable langchain debug logs", default=0
    )


    langchain_hub_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for LangChain Hub"
    )
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangChain tracing v2"
    )
    llm_embedding_model_name: str = Field(
        default="text-embedding-3-large",
        description="The embedding model to use"
    )

    # LangChain Integration Settings
    langchain_endpoint: str = Field(
        env="LANGCHAIN_ENDPOINT",
        description="langchain endpoint",
        default="https://api.smith.langchain.com"
    )
    langchain_hub_api_url: str = Field(
        env="LANGCHAIN_HUB_API_URL",
        description="langchain hub api url for langsmith",
        default="https://api.hub.langchain.com",
    )
    langchain_project: str = Field(
        env="LANGCHAIN_PROJECT",
        description="langsmith project name",
        default="democracy_exe"
    )

    tavily_api_key: SecretStr = Field(
        env="TAVILY_API_KEY",
        description="Tavily API key",
        default=SecretStr("")
    )
    brave_search_api_key: SecretStr = Field(
        env="BRAVE_SEARCH_API_KEY",
        description="Brave Search API key",
        default=SecretStr("")
    )
    unstructured_api_key: SecretStr = Field(
        env="UNSTRUCTURED_API_KEY",
        description="unstructured api key",
        default=SecretStr("")
    )
    unstructured_api_url: str = Field(
        env="UNSTRUCTURED_API_URL",
        description="unstructured api url",
        default="https://api.unstructured.io/general/v0/general",
    )

    debug_aider: bool = Field(
        env="DEBUG_AIDER",
        description="debug tests stuff written by aider",
        default=False
    )
    debug_langgraph_studio: bool = Field(env="DEBUG_LANGGRAPH_STUDIO", description="enable langgraph studio debug", default=False)

    python_fault_handler: bool = Field(
        env="PYTHONFAULTHANDLER",
        description="enable fault handler",
        default=False
    )

    editor: str = Field(
        env="EDITOR",
        description="EDITOR",
        default="vim"
    )
    visual: str = Field(
        env="VISUAL",
        description="VISUAL",
        default="vim"
    )
    git_editor: str = Field(
        env="GIT_EDITOR",
        description="GIT_EDITOR",
        default="vim"
    )

    # tweetpik_api_key: SecretStr = Field(
    #     env="TWEETPIK_API_KEY",
    #     description="TweetPik API key",
    #     default=SecretStr("")
    # )
    # tweetpik_authorization: SecretStr = Field(
    #     env="TWEETPIK_AUTHORIZATION",
    #     description="TweetPik authorization",
    #     default=SecretStr("")
    # )
    # tweetpik_bucket_id: str = Field(
    #     env="TWEETPIK_BUCKET_ID",
    #     description="TweetPik bucket ID",
    #     default="323251495115948625"
    # )
    # tweetpik_theme: str = Field(
    #     env="TWEETPIK_THEME",
    #     description="Theme for tweet screenshots",
    #     default="dim"
    # )
    # tweetpik_dimension: str = Field(
    #     env="TWEETPIK_DIMENSION",
    #     description="Dimension for tweet screenshots",
    #     default="instagramFeed"
    # )

    tool_allowlist: list[str] = ["tavily_search", "magic_function"]
    extension_allowlist: list[str] = ["democracy_exe.chatbot.cogs.twitter"]
    tavily_search_max_results: int = 3

    agent_type: Literal["plan_and_execute", "basic", "advanced", "adaptive_rag"] = Field(
        env="AGENT_TYPE",
        description="Type of agent to use",
        default="adaptive_rag"
    )

    llm_memory_type: str = Field(
        env="LLM_MEMORY_TYPE",
        description="Type of memory to use",
        default="memorysaver"
    )
    llm_memory_enabled: bool = Field(
        env="LLM_MEMORY_ENABLED",
        description="Enable memory",
        default=True
    )
    llm_human_loop_enabled: bool = Field(
        env="LLM_HUMAN_LOOP_ENABLED",
        description="Enable human loop",
        default=False
    )

    text_chunk_size: int = 2000
    text_chunk_overlap: int = 200
    text_splitter: str = "{}"  # custom splitter settings

    qa_completion_llm: str = """{
        "_type": "openai-chat",
        "model_name": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 1000,
        "verbose": true
    }"""
    qa_followup_llm: str = """{
        "_type": "openai-chat",
        "model_name": "gpt-4o-mini",
        "temperature": 0,
        "max_tokens": 200,
        "verbose": true
    }"""
    summarize_llm: str = """{
        "_type": "openai-chat",
        "model_name": "gpt-4o",
        "temperature": 0,
        "max_tokens": 2000
    }"""


    debug: bool = True
    log_pii: bool = True

    sentry_dsn: SecretStr = Field(
        default=SecretStr(""),
        description="Sentry DSN"
    )
    enable_sentry: bool = False

    changelogs_github_api_token: SecretStr = Field(
        env="CHANGELOGS_GITHUB_API_TOKEN",
        description="GitHub API token for Changelogs",
        default=SecretStr("")
    )

    gemini_api_key: SecretStr = Field(
        env="GEMINI_API_KEY",
        description="gemini api key",
        default=SecretStr("")
    )


    default_dropbox_folder: str = "/cerebro_downloads"
    dropbox_cerebro_app_key: SecretStr = Field(
        env="DROPBOX_CEREBRO_APP_KEY",
        description="Dropbox Cerebro App Key",
        default=SecretStr("")
    )
    dropbox_cerebro_app_secret: SecretStr = Field(
        env="DROPBOX_CEREBRO_APP_SECRET",
        description="Dropbox Cerebro App Secret",
        default=SecretStr("")
    )
    dropbox_cerebro_token: SecretStr = Field(
        env="DROPBOX_CEREBRO_TOKEN",
        description="Dropbox Cerebro Token",
        default=SecretStr("")
    )
    dropbox_cerebro_oauth_access_token: SecretStr = Field(
        env="DROPBOX_CEREBRO_OAUTH_ACCESS_TOKEN",
        description="Dropbox Cerebro OAuth Access Token",
        default=SecretStr("")
    )

    firecrawl_api_key: SecretStr = Field(
        env="FIRECRAWL_API_KEY",
        description="Firecrawl API key",
        default=SecretStr("")
    )

    # Model-specific settings
    max_tokens: int = Field(
        env="MAX_TOKENS",
        description="Maximum number of tokens for the model",
        default=900
    )
    max_retries: int = Field(
        env="MAX_RETRIES",
        description="Maximum number of retries for API calls",
        default=9
    )

    chroma_host: str = "localhost"
    chroma_port: str = "9010"
    enable_chroma: bool = True


    active_memory_file: str = Field(
        env="ACTIVE_MEMORY_FILE",
        description="Path to the active memory JSON file",
        default="./active_memory.json",
    )
    add_start_index: bool = Field(
        env="ADD_START_INDEX", description="Whether to add start index to text chunks", default=False
    )

    # Autocrop timeouts
    autocrop_download_timeout: int = Field(
        env="AUTOCROP_DOWNLOAD_TIMEOUT",
        description="Timeout in seconds for downloading images in autocrop",
        default=30
    )
    autocrop_processing_timeout: int = Field(
        env="AUTOCROP_PROCESSING_TIMEOUT",
        description="Timeout in seconds for processing images in autocrop",
        default=60
    )

    bot_name: str = "DemocracyExeAI"

    chunk_size: int = Field(env="CHUNK_SIZE", description="Size of each text chunk", default=1000)
    chunk_overlap: int = Field(env="CHUNK_OVERLAP", description="Overlap between text chunks", default=200)

    dataset_name: str = Field(
        env="DATASET_NAME", description="Name of the dataset to use for evaluation", default="Climate Change Q&A"
    )
    default_search_kwargs: dict[str, int] = Field(
        env="DEFAULT_SEARCH_KWARGS",
        description="Default arguments for similarity search",
        default_factory=lambda: {"k": 2},
    )

    localfilestore_root_path: str = Field(
        env="LOCALFILESTORE_ROOT_PATH", description="root path for local file store", default="./local_file_store"
    )
    log_level: int = 10 # logging.DEBUG
    thirdparty_lib_loglevel: str = "INFO"

# Variables for Postgres/pgvector
    pgvector_driver: str = Field(
        env="PGVECTOR_DRIVER",
        description="The database driver to use for pgvector (e.g., psycopg)",
        default="psycopg",
    )
    pgvector_host: str = Field(
        env="PGVECTOR_HOST",
        description="The hostname or IP address of the pgvector database server",
        default="localhost",
    )
    pgvector_port: int = Field(
        env="PGVECTOR_PORT",
        description="The port number of the pgvector database server",
        default=6432,
    )
    pgvector_database: str = Field(
        env="PGVECTOR_DATABASE",
        description="The name of the pgvector database",
        default="langchain",
    )
    pgvector_user: str = Field(
        env="PGVECTOR_USER",
        description="The username to connect to the pgvector database",
        default="langchain",
    )
    pgvector_password: SecretStr = Field(
        env="PGVECTOR_PASSWORD",
        description="The password to connect to the pgvector database",
        default="langchain",
    )
    pgvector_pool_size: int = Field(
        env="PGVECTOR_POOL_SIZE",
        description="The size of the connection pool for the pgvector database",
        default=10,
    )
    pgvector_dsn_uri: str = Field(
        env="PGVECTOR_DSN_URI",
        description="optional DSN URI, if set other pgvector_* settings are ignored",
        default="",
    )
    provider: str = Field(env="PROVIDER", description="AI provider (openai or anthropic)", default="openai")
    question_to_ask: str = Field(
        env="QUESTION_TO_ASK",
        description="Question to ask for evaluation",
        default="What is the main cause of climate change?",
    )
    retry_stop_after_attempt: int = 3
    retry_wait_exponential_multiplier: int | float = 2
    retry_wait_exponential_max: int | float = 5
    retry_wait_exponential_min: int | float = 1
    retry_wait_fixed: int | float = 15

    scratch_pad_dir: str = Field(
        env="SCRATCH_PAD_DIR",
        description="Directory for scratch pad files",
        default="./scratchpad",
    )

    sklearn_persist_path: str = Field(
        env="SKLEARN_PERSIST_PATH",
        description="Path to persist the SKLearn vector store",
        default="./db.db",
    )
    sklearn_serializer: Literal["json", "bson", "parquet"] = Field(
        env="SKLEARN_SERIALIZER",
        description="Serializer for the SKLearn vector store",
        default="json",
    )
    sklearn_metric: str = Field(
        env="SKLEARN_METRIC",
        description="Metric for the SKLearn vector store",
        default="cosine",
    )
    # Summarization
    summ_default_chain: str = "stuff"
    summ_token_splitter: int = 4000
    summ_token_overlap: int = 500


    tweetpik_api_key: SecretStr = Field(env="TWEETPIK_API_KEY", description="TweetPik API key", default=SecretStr(""))

    tweetpik_authorization: SecretStr = Field(env="TWEETPIK_AUTHORIZATION", description="TweetPik authorization", default=SecretStr(""))
    tweetpik_bucket_id: str = Field(env="TWEETPIK_BUCKET_ID", description="TweetPik bucket ID", default="323251495115948625")
    # change the background color of the tweet screenshot
    tweetpik_background_color: str = "#ffffff"

    # Theme and dimension settings
    tweetpik_theme: str = Field(env="TWEETPIK_THEME", description="Theme for tweet screenshots", default="dim")
    tweetpik_dimension: str = Field(env="TWEETPIK_DIMENSION", description="Dimension for tweet screenshots", default="instagramFeed")

    # Color settings
    tweetpik_background_color: str = Field(env="TWEETPIK_BACKGROUND_COLOR", description="Background color for tweet screenshots", default="#15212b")
    tweetpik_text_primary_color: str = Field(env="TWEETPIK_TEXT_PRIMARY_COLOR", description="Primary text color", default="#FFFFFF")
    tweetpik_text_secondary_color: str = Field(env="TWEETPIK_TEXT_SECONDARY_COLOR", description="Secondary text color", default="#8899a6")
    tweetpik_link_color: str = Field(env="TWEETPIK_LINK_COLOR", description="Color for links and mentions", default="#1b95e0")
    tweetpik_verified_icon_color: str = Field(env="TWEETPIK_VERIFIED_ICON_COLOR", description="Color for verified badge", default="#1b95e0")

    # Display settings
    tweetpik_display_verified: str = Field(env="TWEETPIK_DISPLAY_VERIFIED", description="Show verified badge", default="default")
    tweetpik_display_metrics: bool = Field(env="TWEETPIK_DISPLAY_METRICS", description="Show metrics (likes, retweets)", default=False)
    tweetpik_display_embeds: bool = Field(env="TWEETPIK_DISPLAY_EMBEDS", description="Show embedded content", default=True)

    # Content settings
    tweetpik_content_scale: float = Field(env="TWEETPIK_CONTENT_SCALE", description="Scale factor for content", default=0.77)
    tweetpik_content_width: int = Field(env="TWEETPIK_CONTENT_WIDTH", description="Width of content in percentage", default=100)

    # any number higher than zero. this value is used in pixels(px) units
    tweetpik_canvas_width: str = "510"
    tweetpik_dimension_ig_feed: str = "1:1"
    tweetpik_dimension_ig_story: str = "9:16"
    tweetpik_display_likes: bool = False
    tweetpik_display_link_preview: bool = True
    tweetpik_display_media_images: bool = True
    tweetpik_display_replies: bool = False
    tweetpik_display_retweets: bool = False
    tweetpik_display_source: bool = True
    tweetpik_display_time: bool = True
    tweetpik_display_verified: bool = True

    # change the link colors used for the links, hashtags and mentions
    tweetpik_link_color: str = "#1b95e0"

    tweetpik_text_primary_color: str = (
        "#000000"  # change the text primary color used for the main text of the tweet and user's name
    )
    tweetpik_text_secondary_color: str = (
        "#5b7083"  # change the text secondary used for the secondary info of the tweet like the username
    )

    # any number higher than zero. this value is representing a percentage
    tweetpik_text_width: str = "100"

    tweetpik_timezone: str = "america/new_york"

    # change the verified icon color
    tweetpik_verified_icon: str = "#1b95e0"

    vector_store_type: Literal["pgvector", "chroma", "pinecone", "sklearn"] = "pgvector"




    @model_validator(mode="before")
    @classmethod
    def pre_update(cls, values: dict[str, Any]) -> dict[str, Any]:
        llm_model_name = values.get("llm_model_name")
        llm_embedding_model_name = values.get("llm_embedding_model_name")
        # print(f"llm_model_name: {llm_model_name}")
        # print(f"llm_embedding_model_name: {llm_embedding_model_name}")
        if llm_model_name:
            values["max_tokens"] = MODEL_CONFIG[llm_model_name]["max_tokens"]
            values["max_output_tokens"] = MODEL_CONFIG[llm_model_name]["max_output_tokens"]
            values["prompt_cost_per_token"] = MODEL_CONFIG[llm_model_name]["prompt_cost_per_token"]
            values["completion_cost_per_token"] = MODEL_CONFIG[llm_model_name]["completion_cost_per_token"]
            if llm_embedding_model_name:
                values["embedding_max_tokens"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]
                values["embedding_model_dimensions"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]
        else:
            llm_model_name = "gpt-4o-mini"
            llm_embedding_model_name = "text-embedding-3-large"
            # print(f"setting default llm_model_name: {llm_model_name}")
            # print(f"setting default llm_embedding_model_name: {llm_embedding_model_name}")
            values["max_tokens"] = MODEL_CONFIG[llm_model_name]["max_tokens"]
            values["max_output_tokens"] = MODEL_CONFIG[llm_model_name]["max_output_tokens"]
            values["prompt_cost_per_token"] = MODEL_CONFIG[llm_model_name]["prompt_cost_per_token"]
            values["completion_cost_per_token"] = MODEL_CONFIG[llm_model_name]["completion_cost_per_token"]
            values["embedding_max_tokens"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]
            values["embedding_model_dimensions"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]

        return values

    @model_validator(mode="after")
    def post_root(self) -> Self:
        redis_path = f"/{self.redis_base}" if self.redis_base is not None else ""
        redis_pass = self.redis_pass if self.redis_pass is not None else None
        redis_user = self.redis_user if self.redis_user is not None else None
        if redis_pass is None and redis_user is None:
            self.redis_url = URL.build(
                scheme="redis",
                host=self.redis_host,
                port=self.redis_port,
                path=redis_path,
            )
        else:
            self.redis_url = URL.build(
                scheme="redis",
                host=self.redis_host,
                port=self.redis_port,
                path=redis_path,
                user=redis_user,
                password=redis_pass.get_secret_value(),
            )

        return self

    @field_validator("monitor_port")
    @classmethod
    def validate_port(cls, v: int, info: Any) -> int:
        """Validate port number is in valid range.

        Args:
            v: Port number to validate
            info: Validation context

        Returns:
            int: Validated port number

        Raises:
            ValidationError: If port number is invalid
        """
        if not 1 <= v <= 65535:
            raise ValidationError(
                "Port number must be between 1 and 65535",
                field_name=info.field_name,
                invalid_value=v,
                context={"valid_range": "1-65535"}
            )
        return v

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float, info: Any) -> float:
        """Validate temperature is in valid range.

        Args:
            v: Temperature value to validate
            info: Validation context

        Returns:
            float: Validated temperature value

        Raises:
            ValidationError: If temperature is invalid
        """
        if not 0.0 <= v <= 2.0:
            raise ValidationError(
                "Temperature must be between 0.0 and 2.0",
                field_name=info.field_name,
                invalid_value=v,
                context={"valid_range": "0.0-2.0"}
            )
        return v

    @field_validator("redis_pass")
    @classmethod
    def validate_redis_password(cls, v: SecretStr | None, info: Any) -> SecretStr | None:
        """Validate Redis password if provided.

        Args:
            v: Redis password to validate
            info: Validation context

        Returns:
            Optional[SecretStr]: Validated Redis password

        Raises:
            SecurityError: If password is invalid
        """
        if v is not None:
            password = v.get_secret_value()
            if len(password) < 8:
                raise SecurityError(
                    "Redis password must be at least 8 characters",
                    setting_name=info.field_name,
                    context={"min_length": 8}
                )
        return v

    @field_validator("llm_retry_delay", "llm_retry_max_delay")
    @classmethod
    def validate_retry_delays(cls, v: int, info: Any) -> int:
        """Validate retry delay values.

        Args:
            v: Delay value to validate
            info: Validation context

        Returns:
            int: Validated delay value

        Raises:
            ValidationError: If delay value is invalid
        """
        if v < 0:
            raise ValidationError(
                "Retry delay must be non-negative",
                field_name=info.field_name,
                invalid_value=v,
                context={"min_value": 0}
            )
        if info.field_name == "llm_retry_max_delay" and v < info.data.get("llm_retry_delay", 0):
            raise ValidationError(
                "Maximum retry delay must be greater than initial delay",
                field_name=info.field_name,
                invalid_value=v,
                context={
                    "initial_delay": info.data.get("llm_retry_delay"),
                    "max_delay": v
                }
            )
        return v

    # @property
    # def redis_url(self) -> RedisDsn:
    #     """Get Redis URL.

    #     Constructs a Redis URL from the configured host, port, and authentication settings.
    #     Handles both plain text and SecretStr password types.

    #     Returns:
    #         RedisDsn: Redis URL with proper authentication if configured
    #     """
    #     auth = ""
    #     if self.redis_user and self.redis_pass:
    #         pass_value = self.redis_pass
    #         if isinstance(pass_value, SecretStr):
    #             pass_value = pass_value.get_secret_value()
    #         auth = f"{self.redis_user}:{pass_value}@"
    #     base = f"/{self.redis_base}" if self.redis_base is not None else ""
    #     return RedisDsn(f"redis://{auth}{self.redis_host}:{self.redis_port}{base}")

    @property
    def postgres_url(self) -> str:
        """Get the PostgreSQL URL.

        Returns:
            str: The PostgreSQL URL
        """
        return f"postgresql+{self.postgres_driver}://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"

    @model_validator(mode="before")
    @classmethod
    def validate_models(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate and update model configurations.

        This validator runs before other validations and updates various settings based on
        the selected models' configurations. It handles both LLM and embedding model settings.

        Args:
            values: Values to validate and update

        Returns:
            dict[str, Any]: Updated values

        Raises:
            ModelConfigError: If model configuration is invalid or missing required settings
        """
        llm_model_name = values.get("llm_model_name")
        llm_embedding_model_name = values.get("llm_embedding_model_name")
        openai_embeddings_model = values.get("openai_embeddings_model")

        # Validate LLM model settings
        if llm_model_name:
            # if llm_model_name not in MODEL_CONFIG:
            #     raise ModelConfigError(
            #         "Invalid model name",
            #         model_name=llm_model_name,
            #         context={
            #             "available_models": list(MODEL_CONFIG.keys())
            #         }
            #     )

            config = MODEL_CONFIG[llm_model_name]
            values["llm_max_tokens"] = config["max_tokens"]
            values["llm_max_output_tokens"] = config["max_output_tokens"]
            values["prompt_cost_per_token"] = config["prompt_cost_per_token"]
            values["completion_cost_per_token"] = config["completion_cost_per_token"]

        # Validate embedding models
        for model_name in [llm_embedding_model_name, openai_embeddings_model]:
            if model_name and model_name not in EMBEDDING_CONFIG:
                # raise ModelConfigError(
                #     "Invalid embedding model name",
                #     model_name=model_name,
                #     context={
                #         "available_models": list(EMBEDDING_CONFIG.keys())
                #     }
                # )
                pass

        # Update embedding dimensions
        if llm_embedding_model_name:
            values["embedding_model_dimensions"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]

        # Validate OpenCommit settings
        oco_model = values.get("oco_model")
        # rich.print(f"oco_model: {oco_model}")
        # if oco_model and oco_model not in MODEL_CONFIG:
        #     raise ModelConfigError(
        #         "Invalid OCO model name",
        #         model_name=oco_model,
        #         context={
        #             "available_models": list(MODEL_CONFIG.keys())
        #         }
        #     )

        return values

    @field_serializer(
        "discord_token",
        "openai_api_key",
        "redis_pass",
        "pinecone_api_key",
        "langchain_api_key",
        "langchain_hub_api_key",
        when_used="json",
    )
    def dump_secret(self, v):
        return v.get_secret_value()


def get_rich_console() -> Console:
    """Get a Rich console instance for formatted output.

    Returns:
        Console: A configured Rich console instance for formatted terminal output
    """
    return Console()

# Global settings instance
aiosettings = AioSettings()
# avoid-global-variables
# In-place reloading
aiosettings.__init__()
