"""Settings for the Discord bot and AI components.

This module contains the settings for the Discord bot and AI components,
using Pydantic for validation.
"""
from __future__ import annotations

from typing import Optional

import rich.console

from pydantic import BaseModel, Field, SecretStr


def get_rich_console() -> rich.console.Console:
    """Get a Rich console instance.

    Returns:
        rich.console.Console: Configured console instance
    """
    return rich.console.Console()


class AioSettings(BaseModel):
    """Settings for the Discord bot and AI components.

    Attributes:
        monitor_host: Host for monitoring server
        monitor_port: Port for monitoring server
        debug_langchain: Enable debug logging for LangChain
        audit_log_send_channel: Channel ID for audit logs
        prefix: Command prefix for the bot
        discord_client_id: Discord client ID
        discord_client_secret: Discord client secret
        discord_token: Discord bot token
        enable_ai: Enable AI features
        rate_limit_rate: Rate limit for commands
        rate_limit_per: Time window for rate limit
        spam_ban_threshold: Number of rate limit violations before ban
        max_queue_size: Maximum size of task queue
        num_workers: Number of worker threads
        max_memory_mb: Maximum memory usage in MB
        max_tasks: Maximum number of concurrent tasks
        max_response_size_mb: Maximum response size in MB
        max_buffer_size_kb: Maximum buffer size in KB
        task_timeout_seconds: Task timeout in seconds
        dev_mode: Enable development mode for additional debugging and error handling
        llm_model_name: Name of the LLM model to use
        llm_embedding_model_name: Name of the embedding model to use
        llm_api_key: API key for LLM service
        llm_api_base: Base URL for LLM service
        llm_api_version: API version for LLM service
        llm_api_type: Type of LLM API
        llm_deployment_name: Name of LLM deployment
        llm_temperature: Temperature for LLM sampling
        llm_max_tokens: Maximum tokens for LLM response
        llm_top_p: Top p for LLM sampling
        llm_frequency_penalty: Frequency penalty for LLM
        llm_presence_penalty: Presence penalty for LLM
        llm_stop: Stop sequences for LLM
        llm_timeout: Timeout for LLM requests
        llm_max_retries: Maximum retries for LLM requests
        llm_retry_delay: Initial delay between retries in seconds
        llm_retry_multiplier: Multiplier for retry delay
        llm_retry_max_delay: Maximum delay between retries in seconds
        llm_retry_codes: HTTP status codes to retry on
        llm_retry_methods: HTTP methods to retry
        llm_retry_statuses: HTTP status codes to retry on
        llm_retry_backoff: Whether to use exponential backoff
        llm_retry_jitter: Whether to add jitter to retry delay
        llm_retry_raise_for_status: Whether to raise for status
        llm_retry_respect_retry_after: Whether to respect Retry-After header
        llm_retry_forcelist: HTTP status codes to force retry on
        llm_retry_allowed_methods: HTTP methods allowed for retry
        llm_retry_connect: Number of connection retries
        llm_retry_read: Number of read retries
        llm_retry_redirect: Number of redirect retries
        llm_retry_status: Number of status retries
        llm_retry_other: Number of other retries
        llm_retry_backoff_factor: Factor to multiply delay by
        llm_retry_raise_on_redirect: Whether to raise on redirect
        llm_retry_raise_on_status: Whether to raise on status
        llm_retry_history: Whether to maintain retry history
        llm_retry_respect_retry_after_header: Whether to respect Retry-After header
        llm_retry_incremental_backoff: Whether to use incremental backoff
        embedding_max_tokens: Maximum tokens for embeddings
        embedding_model_dimensions: Dimensions of embedding model
    """
    # Monitor settings
    monitor_host: str = Field(default="localhost")
    monitor_port: int = Field(default=50102)
    debug_langchain: bool = Field(default=True)
    audit_log_send_channel: str = Field(default="")

    # Development settings
    dev_mode: bool = Field(
        default=False,
        description="Enable development mode for additional debugging and error handling"
    )

    # Bot settings
    prefix: str = Field(default="?")
    discord_client_id: str = Field(default="")
    discord_client_secret: SecretStr = Field(default=SecretStr(""))
    discord_token: SecretStr = Field(default=SecretStr(""))
    enable_ai: bool = Field(default=True)
    rate_limit_rate: int = Field(default=10)
    rate_limit_per: float = Field(default=12.0)
    spam_ban_threshold: int = Field(default=5)
    max_queue_size: int = Field(default=1000)
    num_workers: int = Field(default=3)

    # Resource management settings
    max_memory_mb: int = Field(default=512)
    max_tasks: int = Field(default=100)
    max_response_size_mb: int = Field(default=8)
    max_buffer_size_kb: int = Field(default=64)
    task_timeout_seconds: int = Field(default=300)

    # LLM settings
    llm_model_name: str = Field(default="gpt-4o-mini")
    llm_embedding_model_name: str = Field(default="text-embedding-3-large")
    llm_api_key: str = Field(default="")
    llm_api_base: str = Field(default="")
    llm_api_version: str = Field(default="")
    llm_api_type: str = Field(default="")
    llm_deployment_name: str = Field(default="")
    llm_temperature: float = Field(default=0.7)
    llm_max_tokens: int = Field(default=1024)
    llm_top_p: float = Field(default=1.0)
    llm_frequency_penalty: float = Field(default=0.0)
    llm_presence_penalty: float = Field(default=0.0)
    llm_stop: list[str] = Field(default_factory=list)
    llm_timeout: int = Field(default=60)
    llm_max_retries: int = Field(default=3)
    llm_retry_delay: int = Field(default=1)
    llm_retry_multiplier: float = Field(default=2.0)
    llm_retry_max_delay: int = Field(default=60)
    llm_retry_codes: list[int] = Field(default_factory=list)
    llm_retry_methods: list[str] = Field(default_factory=list)
    llm_retry_statuses: list[int] = Field(default_factory=list)
    llm_retry_backoff: bool = Field(default=True)
    llm_retry_jitter: bool = Field(default=True)
    llm_retry_raise_for_status: bool = Field(default=True)
    llm_retry_respect_retry_after: bool = Field(default=True)
    llm_retry_forcelist: list[int] = Field(default_factory=list)
    llm_retry_allowed_methods: list[str] = Field(default_factory=list)
    llm_retry_connect: int = Field(default=3)
    llm_retry_read: int = Field(default=3)
    llm_retry_redirect: int = Field(default=3)
    llm_retry_status: int = Field(default=3)
    llm_retry_other: int = Field(default=3)
    llm_retry_backoff_factor: float = Field(default=0.3)
    llm_retry_raise_on_redirect: bool = Field(default=True)
    llm_retry_raise_on_status: bool = Field(default=True)
    llm_retry_history: bool = Field(default=True)
    llm_retry_respect_retry_after_header: bool = Field(default=True)
    llm_retry_incremental_backoff: bool = Field(default=True)

    # Embedding settings
    embedding_max_tokens: int = Field(default=1024)
    embedding_model_dimensions: int = Field(default=1024)

# Global settings instance
aiosettings = AioSettings()
