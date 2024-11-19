# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import time
import uuid

from functools import lru_cache

import _schemas as schemas
import langsmith

from langchain_core.runnables import RunnableConfig
from langchain_fireworks import FireworksEmbeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from pinecone import Pinecone, ServerlessSpec
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
    pc: Pinecone = get_or_create_index()
    # pc: Pinecone = Pinecone(api_key=aiosettings.pinecone_api_key.get_secret_value()) # pylint: disable=no-member
    # pc: Pinecone = Pinecone(api_key=aiosettings.pinecone_api_key.get_secret_value(), environment=aiosettings.pinecone_env) # pylint: disable=no-member
    return pc



def get_or_create_index() -> Pinecone.Index:
    """Get or create a Pinecone index instance using settings from aiosettings.

    This function checks if the index exists, creates it if it doesn't, and returns
    the index instance. It waits for the index to be ready before returning.

    Returns:
        Pinecone.Index: A configured Pinecone index instance.

    Note:
        If the index doesn't exist, it will be created with dimension=3072 and
        metric="cosine" in the us-east-1 region.
    """
    pc: Pinecone = Pinecone(api_key=aiosettings.pinecone_api_key.get_secret_value()) # pylint: disable=no-member
    index_name: str = aiosettings.pinecone_index_name

    existing_indexes: list[str] = [index_info["name"] for index_info in pc.list_indexes()]

    logger.info(f"Existing indexes: {existing_indexes}")

    if index_name not in existing_indexes:
        logger.info(f"Creating index: {index_name} with dimension=3072 and metric=cosine in us-east-1")
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(index_name)



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
            model=configurable.get("model", "gpt-4o"),
            thread_id=configurable["thread_id"],
            user_id=configurable["user_id"],
        ),
    }


@lru_cache
def get_embeddings(model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> FireworksEmbeddings:
    if model_name == "nomic-ai/nomic-embed-text-v1.5":
        return FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
    elif model_name == "text-embedding-3-large":
        return OpenAIEmbeddings(model="text-embedding-3-large")
    return FireworksEmbeddings(model=model_name)


__all__ = ["ensure_configurable"]
