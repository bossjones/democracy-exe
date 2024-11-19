from __future__ import annotations

import os

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Optional

from langchain_core.runnables import RunnableConfig
from loguru import logger


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    user_id: str = "default-user"
    model: str = "gpt-4o"
    provider: str = "openai"
    # for debounding memory creation
    delay: int = 60

    @classmethod
    def from_runnable_config(
        cls, config: RunnableConfig | None = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        logger.error(f"Configuration: {values}")
        return cls(**{k: v for k, v in values.items() if v})
