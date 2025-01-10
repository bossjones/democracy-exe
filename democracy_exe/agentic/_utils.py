# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import time
import uuid

from functools import lru_cache
from typing import TYPE_CHECKING, Union

import langsmith
import structlog

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableConfig
from langchain_fireworks import FireworksEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


logger = structlog.get_logger(__name__)
from pinecone import Pinecone, ServerlessSpec

import democracy_exe.agentic._schemas as schemas

from democracy_exe.aio_settings import aiosettings


if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    # Type alias for chat models
    ChatModelLike = Union[ChatOpenAI, ChatAnthropic]

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

    Note:
        Uses UUID v5 which generates a deterministic UUID based on a namespace and name,
        ensuring the same user_id always generates the same thread_id.
    """
    # Use DNS namespace as a stable namespace for UUID generation
    # UUID v5 requires a namespace UUID and a name to generate a deterministic UUID
    namespace: uuid.UUID = uuid.NAMESPACE_DNS  # You can choose a different namespace if appropriate

    # Create a unique name string by prefixing the user_id with "USER:"
    # This helps avoid potential collisions with other UUID generation in the system
    name: str = f"USER:{user_id}"

    # Generate a UUID v5 using the namespace and name
    # UUID v5 uses SHA-1 hashing to create a deterministic UUID
    generated_uuid: uuid.UUID = uuid.uuid5(namespace, name)

    # Log the components and result for debugging purposes
    logger.info(f"namespace: {namespace}")
    logger.info(f"name: {name}")
    logger.info(f"Generated fake thread ID: {generated_uuid}")

    # Convert the UUID to a string and return it
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



@langsmith.traceable  # Decorator to enable tracing of this function in LangSmith
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
    # Check if we're running in terminal mode vs discord mode
    if aiosettings.chatbot_type == "terminal":
        # For terminal mode, use a default user ID of 1
        user_id: int = 1
        # Generate a deterministic thread ID based on the user ID
        thread_id: str = get_fake_thread_id(user_id=user_id)

        # Create a configurable dict with the fake thread_id and user_id, or use existing config if provided
        configurable: dict[str, str | int] = config.get("configurable", {"thread_id": thread_id, "user_id": user_id})
        # Log the terminal configuration for debugging
        logger.info(f"Using terminal config: {configurable}")
    else:
        # For discord mode, get the configurable dict from the config, or use empty dict if not provided
        configurable: dict = config.get("configurable", {})
        # Log the discord configuration for debugging
        logger.info(f"Using discord config: {configurable}")

    # Return a merged dictionary containing:
    # 1. All key/values from the configurable dict
    # 2. A new GraphConfig with:
    #    - delay: use value from configurable or default to _DEFAULT_DELAY
    #    - model: use value from configurable or default to "gpt-4o"
    #    - thread_id: required value from configurable
    #    - user_id: required value from configurable
    return {
        **configurable,  # Spread all existing configurable key/values
        **schemas.GraphConfig(  # Create and spread a new GraphConfig with defaults
            delay=configurable.get("delay", _DEFAULT_DELAY),  # Get delay or use default
            model=configurable.get("model", "gpt-4o"),  # Get model or use default
            thread_id=configurable["thread_id"],  # Required field
            user_id=configurable["user_id"],  # Required field
        ),
    }


@lru_cache(maxsize=32)  # Cache the results of this function to avoid recreating embedding models unnecessarily
def get_embeddings(model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> FireworksEmbeddings:
    """Get an embedding model instance based on the model name.

    Args:
        model_name (str): Name of the embedding model to use. Defaults to "nomic-ai/nomic-embed-text-v1.5".

    Returns:
        FireworksEmbeddings | OpenAIEmbeddings: An instance of the specified embedding model.

    Note:
        The function is cached using @lru_cache to avoid recreating the same model multiple times.
        Currently supports:
        - nomic-ai/nomic-embed-text-v1.5 (default, uses FireworksEmbeddings)
        - text-embedding-3-large (uses OpenAIEmbeddings)
        - Any other model name will use FireworksEmbeddings
    """
    # If using the default Nomic AI model
    if model_name == "nomic-ai/nomic-embed-text-v1.5":
        # Return a FireworksEmbeddings instance configured for the Nomic AI model
        return FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
    # If using OpenAI's text-embedding-3-large model
    elif model_name == "text-embedding-3-large":
        # Return an OpenAIEmbeddings instance configured for the text-embedding-3-large model
        return OpenAIEmbeddings(model="text-embedding-3-large")
    # For any other model name
    return FireworksEmbeddings(model=model_name)  # Use FireworksEmbeddings with the specified model name


@lru_cache(maxsize=32)
def get_chat_model(
    model_name: str,
    model_provider: str = "openai",
    temperature: float = 0.0
) -> ChatModelLike:
    """Get a chat model instance based on the model name and provider.

    Args:
        model_name (str): Name of the model to use
        model_provider (str): Provider of the model ('openai' or 'anthropic'). Defaults to 'openai'.
        temperature (float): Temperature for model generation. Defaults to 0.0.

    Returns:
        ChatModelLike: An instance of either ChatOpenAI or ChatAnthropic.

    Note:
        The function is cached using @lru_cache to avoid recreating the same model multiple times.
        Currently supports:
        - OpenAI models (using ChatOpenAI)
        - Anthropic models (using ChatAnthropic)
    """
    if model_provider == "anthropic":
        return ChatAnthropic(model=model_name, temperature=temperature)
    return ChatOpenAI(model=model_name, temperature=temperature)


__all__ = ["ensure_configurable", "get_embeddings", "get_chat_model"]
