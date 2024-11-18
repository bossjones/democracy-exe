# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from functools import lru_cache

import langsmith

from langchain_core.runnables import RunnableConfig
from langchain_fireworks import FireworksEmbeddings
from pinecone import Pinecone

from democracy_exe.agentic import _schemas as schemas
from democracy_exe.aio_settings import aiosettings


_DEFAULT_DELAY = 60  # seconds


def get_index():
    pc = Pinecone(api_key=aiosettings.pinecone_api_key)
    return pc.Index(aiosettings.pinecone_index_name)


@langsmith.traceable
def ensure_configurable(config: RunnableConfig) -> schemas.GraphConfig:
    """Merge the user-provided config with default values."""
    configurable = config.get("configurable", {})
    return {
        **configurable,
        **schemas.GraphConfig(
            delay=configurable.get("delay", _DEFAULT_DELAY),
            model=configurable.get("model", aiosettings.chat_model),
            thread_id=configurable["thread_id"],
            user_id=configurable["user_id"],
        ),
    }


@lru_cache
def get_embeddings():
    return FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")


__all__ = ["ensure_configurable"]
