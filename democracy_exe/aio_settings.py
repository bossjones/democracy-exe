# pylint: disable=no-name-in-module
# pylint: disable=no-member
# pyright: reportInvalidTypeForm=false
# pyright: reportUndefinedVariable=false

"""Settings for the Discord bot and AI components.

This module contains the settings for the Discord bot and AI components,
using Pydantic for validation and environment variable configuration.

Key Features:
- Environment variable configuration with prefix DEMOCRACY_EXE_CONFIG_
- Automatic model token limit updates based on model selection
- Comprehensive retry configuration for LLM calls
- Integrated Redis and PostgreSQL support
- QA and summarization configuration

Example:
    ```python
    settings = AioSettings()
    # Override from environment:
    # export DEMOCRACY_EXE_CONFIG_LLM_MODEL_NAME=gpt-4-0613
    ```
"""
from __future__ import annotations

import enum
import pathlib

from datetime import timedelta, timezone
from typing import Any, Dict, List, Literal, Optional, Union, cast

from pydantic import Field, PostgresDsn, RedisDsn, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from democracy_exe import __version__


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


# Older model configurations
_OLDER_MODEL_CONFIG: dict[str, dict[str, int | float]] = {
    "gpt-4": {
        "max_tokens": 8192,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00003,
        "completion_cost_per_token": 0.00006
    },
    "gpt-4-0613": {
        "max_tokens": 8192,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00003,
        "completion_cost_per_token": 0.00006
    },
    "gpt-4-32k": {
        "max_tokens": 32768,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00006,
        "completion_cost_per_token": 0.00012
    },
    "gpt-3.5-turbo": {
        "max_tokens": 4096,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000001,
        "completion_cost_per_token": 0.000002
    },
    "gpt-3.5-turbo-16k": {
        "max_tokens": 16384,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.000003,
        "completion_cost_per_token": 0.000004
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
    "gpt-4-turbo-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003
    },
    "gpt-4-vision-preview": {
        "max_tokens": 128000,
        "max_output_tokens": 4096,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003
    },
    "claude-3-opus-20240229": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.000015,
        "completion_cost_per_token": 0.000007
    },
    "claude-3-sonnet-20240229": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.000003,
        "completion_cost_per_token": 0.0000015
    },
    "claude-3-haiku-20240307": {
        "max_tokens": 2048,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.0000025,
        "completion_cost_per_token": 0.000001
    },
    "gpt-4o": {
        "max_tokens": 128000,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003
    },
    "gpt-4o-mini-2024-07-18": {
        "max_tokens": 900,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.000005,
        "completion_cost_per_token": 0.000015
    },
    "gpt-4o-2024-08-06": {
        "max_tokens": 128000,
        "max_output_tokens": 16384,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003
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
    "gpt4": "gpt-4",
    "gpt4-32k": "gpt-4-32k",
    "gpt4-turbo": "gpt-4-turbo-preview",
    "gpt4-vision": "gpt-4-vision-preview",
    "gpt35": "gpt-3.5-turbo",
    "gpt35-16k": "gpt-3.5-turbo-16k",
    "claude3-opus": "claude-3-opus-20240229",
    "claude3-sonnet": "claude-3-sonnet-20240229",
    "claude3-haiku": "claude-3-haiku-20240307",
    "gpt4o": "gpt-4o-2024-08-06",
    "gpt4o-mini": "gpt-4o-mini-2024-07-18"
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

# Create model zoo for validation
MODEL_ZOO: set[str] = set(MODEL_CONFIG.keys()) | set(EMBEDDING_CONFIG.keys()) | {"text-embedding-3-small", "text-embedding-3-large"}

# Embedding model dimensions for different models
EMBEDDING_MODEL_DIMENSIONS_DATA: dict[str, int] = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 1024
}

TIMEZONE = timezone(timedelta(hours=8), name='Asia/Kuala_Lumpur')

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
        default=SecretStr(""),
        description="OpenAI API key for OpenCommit"
    )
    oco_tokens_max_input: int = Field(
        default=4096,
        description="Maximum input tokens for OpenCommit"
    )
    oco_tokens_max_output: int = Field(
        default=500,
        description="Maximum output tokens for OpenCommit"
    )
    oco_model: str = Field(
        default="gpt-4o",
        description="Model to use for OpenCommit"
    )
    oco_language: str = Field(
        default="en",
        description="Language for OpenCommit messages"
    )
    oco_prompt_module: str = Field(
        default="conventional-commit",
        description="Prompt module for OpenCommit"
    )
    oco_ai_provider: str = Field(
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
    pythonasynciodebug: int = Field(
        default=1,
        description="Enable asyncio debug mode"
    )

    # Bot settings
    prefix: str = Field(
        default="?",
        description="Command prefix for the Discord bot"
    )
    discord_client_id: str = Field(
        default="",
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
        default=False,
        description="Enable Python development mode"
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
    MODEL_CONFIG: dict[str, dict[str, int | float]] = {**_OLDER_MODEL_CONFIG, **_NEWER_MODEL_CONFIG}
    MODEL_POINT: dict[str, str] = {
        "gpt4": "gpt-4",
        "gpt4-32k": "gpt-4-32k",
        "gpt35": "gpt-3.5-turbo",
        "gpt35-16k": "gpt-3.5-turbo-16k",
        "gpt4-turbo": "gpt-4-turbo-preview",
        "gpt4-vision": "gpt-4-vision-preview",
        "claude3-opus": "claude-3-opus-20240229",
        "claude3-sonnet": "claude-3-sonnet-20240229",
        "claude3-haiku": "claude-3-haiku-20240307",
        "gpt4o": "gpt-4o-2024-08-06",
        "gpt4o-mini": "gpt-4o-mini-2024-07-18"
    }
    MODEL_ZOO: set[str] = set(MODEL_CONFIG.keys())
    EMBEDDING_CONFIG: dict[str, dict[str, int | float]] = {**_OLDER_EMBEDDING_CONFIG, **_NEWER_EMBEDDING_CONFIG}
    EMBEDDING_MODEL_DIMENSIONS_DATA: dict[str, int] = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 1024
    }

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
        default=SecretStr(""),
        description="Pinecone API key"
    )
    cohere_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Cohere API key"
    )
    anthropic_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Anthropic API key"
    )

    chat_history_buffer: int = Field(
        default=10,
        description="Number of messages to keep in chat history"
    )

    chat_model: str = Field(
        default="gpt-4o-mini",
        description="The chat model to use"
    )

    eval_max_concurrency: int = Field(
        default=4,
        description="Maximum number of concurrent evaluation tasks"
    )

    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for Groq"
    )

    langchain_api_key: SecretStr = Field(
        default=SecretStr(""),
        description="API key for LangChain"
    )

    langchain_debug_logs: bool = Field(
        default=False,
        description="Enable LangChain debug logging"
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

    @field_validator("monitor_port")
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
            # pyright: reportAttributeAccessIssue=false
            password = v.get_secret_value()
            if len(password) < 8:
                raise SecurityError(
                    "Redis password must be at least 8 characters",
                    setting_name=info.field_name,
                    context={"min_length": 8}
                )
        return v

    @field_validator("llm_retry_delay", "llm_retry_max_delay")
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

    @property
    def redis_url(self) -> RedisDsn:
        """Get Redis URL.

        Constructs a Redis URL from the configured host, port, and authentication settings.
        Handles both plain text and SecretStr password types.

        Returns:
            RedisDsn: Redis URL with proper authentication if configured
        """
        auth = ""
        if self.redis_user and self.redis_pass:
            # pyright: reportAttributeAccessIssue=false
            pass_value = self.redis_pass
            if isinstance(pass_value, SecretStr):
                pass_value = pass_value.get_secret_value()
            auth = f"{self.redis_user}:{pass_value}@"
        base = f"/{self.redis_base}" if self.redis_base is not None else ""
        return RedisDsn(f"redis://{auth}{self.redis_host}:{self.redis_port}{base}")

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
            if llm_model_name not in MODEL_CONFIG:
                raise ModelConfigError(
                    "Invalid model name",
                    model_name=llm_model_name,
                    context={
                        "available_models": list(MODEL_CONFIG.keys())
                    }
                )

            config = MODEL_CONFIG[llm_model_name]
            values["llm_max_tokens"] = config["max_tokens"]
            values["llm_max_output_tokens"] = config["max_output_tokens"]
            values["prompt_cost_per_token"] = config["prompt_cost_per_token"]
            values["completion_cost_per_token"] = config["completion_cost_per_token"]

        # Validate embedding models
        for model_name in [llm_embedding_model_name, openai_embeddings_model]:
            if model_name and model_name not in EMBEDDING_CONFIG:
                raise ModelConfigError(
                    "Invalid embedding model name",
                    model_name=model_name,
                    context={
                        "available_models": list(EMBEDDING_CONFIG.keys())
                    }
                )

        # Update embedding dimensions
        if llm_embedding_model_name:
            values["embedding_model_dimensions"] = EMBEDDING_MODEL_DIMENSIONS_DATA[llm_embedding_model_name]

        # Validate OpenCommit settings
        oco_model = values.get("oco_model")
        if oco_model and oco_model not in MODEL_CONFIG:
            raise ModelConfigError(
                "Invalid OCO model name",
                model_name=oco_model,
                context={
                    "available_models": list(MODEL_CONFIG.keys())
                }
            )

        return values

# Global settings instance
aiosettings = AioSettings()
