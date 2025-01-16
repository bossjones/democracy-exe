# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import os
import tempfile
import time
import uuid

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import langsmith
import structlog

from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.runnables import RunnableConfig
from langchain_fireworks import FireworksEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import democracy_exe.agentic._schemas as schemas

from democracy_exe.aio_settings import aiosettings


if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.embeddings import Embeddings
    from langchain_openai import ChatOpenAI

    # Type alias for chat models
    ChatModelLike = Union[ChatOpenAI, ChatAnthropic]

logger = structlog.get_logger(__name__)

def get_or_create_sklearn_index(
    embeddings: Embeddings,
    persist_path: str | Path | None = None,
    serializer: str = "parquet"
) -> SKLearnVectorStore:
    """Get or create a SKLearnVectorStore instance.

    This function checks if a vector store exists at the specified path,
    loads it if it does, or creates a new one if it doesn't.

    Args:
        embeddings (Embeddings): The embeddings model to use
        persist_path (Optional[Union[str, Path]], optional): Path to save/load the vector store.
            If None, creates one in a temporary directory. Defaults to None.
        serializer (str, optional): Serialization format ('json', 'bson', or 'parquet').
            Defaults to "parquet".

    Returns:
        SKLearnVectorStore: A configured vector store instance.

    Note:
        If persist_path is not provided, creates a temporary file with format:
        /tmp/sklearn_vector_store_{uuid}.{serializer}
    """
    # If no persist_path provided, create one in temp directory
    if persist_path is None:
        temp_dir = tempfile.gettempdir()
        persist_path = os.path.join(
            temp_dir,
            f"sklearn_vector_store_{uuid.uuid4()}.{serializer}"
        )
        logger.info(f"Created temporary persist path: {persist_path}")

    # Convert to Path object for easier handling
    persist_path = Path(persist_path)

    # Check if vector store exists at path
    if persist_path.exists():
        logger.info(f"Loading existing vector store from: {persist_path}")
        return SKLearnVectorStore(
            embedding=embeddings,
            persist_path=str(persist_path),
            serializer=serializer
        )

    # Create new vector store if doesn't exist
    logger.info(f"Creating new vector store at: {persist_path}")
    vector_store = SKLearnVectorStore(
        embedding=embeddings,
        persist_path=str(persist_path),
        serializer=serializer
    )

    # Ensure directory exists
    persist_path.parent.mkdir(parents=True, exist_ok=True)

    # Persist empty vector store
    vector_store.persist()

    return vector_store

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


# @langsmith.traceable  # Decorator to enable tracing of this function in LangSmith
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


__all__ = ["ensure_configurable", "get_embeddings", "get_chat_model", "get_or_create_sklearn_index"]
