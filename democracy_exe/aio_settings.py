"""Async settings for the application."""
from __future__ import annotations

import os

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class AioSettings(BaseModel):
    """Async settings for the application."""

    # Monitor settings
    monitor_host: str = Field(default="localhost")
    monitor_port: int = Field(default=50102)

    # Debug settings
    debug_langchain: bool = Field(default=True)

    # Audit log settings
    audit_log_send_channel: str = Field(default="")

    # Bot settings
    prefix: str = Field(default="?")
    bot_token: str = Field(default="")
    bot_owner_id: int = Field(default=0)
    bot_guild_id: int = Field(default=0)
    bot_channel_id: int = Field(default=0)
    bot_role_id: int = Field(default=0)

    # LLM settings
    llm_model_name: str | None = Field(default=None)
    llm_embedding_model_name: str | None = Field(default=None)
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

    # Cost settings
    prompt_cost_per_token: float = Field(default=1.5e-7)
    completion_cost_per_token: float = Field(default=6e-7)

    # Embedding settings
    embedding_max_tokens: int = Field(default=1024)
    embedding_model_dimensions: int = Field(default=1024)

    # Resource management settings
    max_memory_mb: int = Field(default=512)  # Maximum memory usage in MB
    max_tasks: int = Field(default=100)  # Maximum number of concurrent tasks
    max_response_size_mb: int = Field(default=1)  # Maximum response size in MB
    max_buffer_size_kb: int = Field(default=64)  # Maximum buffer size in KB
    task_timeout_seconds: int = Field(default=30)  # Task timeout in seconds

    def __init__(self, **data: Any) -> None:
        """Initialize settings.

        Args:
            **data: Settings data
        """
        super().__init__(**data)

        # Set default model names if not provided
        if self.llm_model_name is None:
            self.llm_model_name = "gpt-4o-mini"
        if self.llm_embedding_model_name is None:
            self.llm_embedding_model_name = "text-embedding-3-large"

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary.

        Returns:
            Dict[str, Any]: Settings dictionary
        """
        return self.model_dump()


# Global settings instance
aiosettings = AioSettings()
