# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import uuid

from functools import lru_cache

import _schemas as schemas
import langsmith

from langchain_core.runnables import RunnableConfig
from langchain_fireworks import FireworksEmbeddings
from loguru import logger
from pinecone import Pinecone
from settings import aiosettings


_DEFAULT_DELAY = 60  # seconds


# def get_fake_user_id_to_uuid(user_id: int = 1) -> str:
#     namespace = uuid.NAMESPACE_DNS  # You can choose a different namespace if appropriate
#     name = f"USER:{user_id}"
#     generated_uuid = uuid.uuid5(namespace, name)
#     logger.info(f"Generated fake user ID: {generated_uuid}")
#     return generated_uuid

def get_fake_thread_id(user_id: int = 1) -> str:
    """Generate a deterministic UUID for a thread based on user ID.

    Args:
        user_id (int): The user ID to generate a thread ID for. Defaults to 1.

    Returns:
        str: A UUID v5 string generated from the user ID.
    """
    namespace: uuid.UUID = uuid.NAMESPACE_DNS  # You can choose a different namespace if appropriate
    name: str = f"USER:{user_id}"
    generated_uuid: uuid.UUID = uuid.uuid5(namespace, name)
    logger.info(f"Generated fake thread ID: {generated_uuid}")
    return str(generated_uuid)

def get_index() -> Pinecone.Index:
    """Get a Pinecone index instance using settings from aiosettings.

    Returns:
        Pinecone.Index: A configured Pinecone index instance.
    """
    pc: Pinecone = Pinecone(api_key=aiosettings.pinecone_api_key)
    return pc.Index(aiosettings.pinecone_index_name)


@langsmith.traceable
def ensure_configurable(config: RunnableConfig) -> schemas.GraphConfig:
    """Merge the user-provided config with default values.

    Args:
        config (RunnableConfig): The configuration object containing user settings.

    Returns:
        schemas.GraphConfig: A merged configuration containing both user-provided and default values.

    Note:
        If chatbot_type is "terminal", it will generate a fake thread_id and user_id.
        Otherwise, it will use the provided discord configuration.
    """
    if aiosettings.chatbot_type == "terminal":
        user_id: int = 1
        thread_id: str = get_fake_thread_id(user_id=user_id)

        configurable: dict[str, str | int] = config.get("configurable", {"thread_id": thread_id, "user_id": user_id})
        logger.info(f"Using terminal config: {configurable}")
    else:
        configurable: dict = config.get("configurable", {})
        logger.info(f"Using discord config: {configurable}")

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
