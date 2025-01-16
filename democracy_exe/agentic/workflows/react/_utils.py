# pyright: reportUninitializedInstanceVariable=false
# pyright: reportUndefinedVariable=false
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import time
import uuid

from functools import lru_cache

import langsmith
import structlog

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

import democracy_exe.agentic.workflows.react._schemas as schemas

from democracy_exe.aio_settings import aiosettings


logger = structlog.get_logger(__name__)

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)


_DEFAULT_DELAY = 60  # seconds


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
    logger.info(f"namespace: {namespace}")
    logger.info(f"name: {name}")
    logger.info(f"Generated fake thread ID: {generated_uuid}")
    return str(generated_uuid)



# @langsmith.traceable
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
def get_embeddings(model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> Embeddings:
    if model_name == "nomic-ai/nomic-embed-text-v1.5":
        from langchain_fireworks import FireworksEmbeddings
        return FireworksEmbeddings(model="nomic-ai/nomic-embed-text-v1.5")
    elif model_name == "text-embedding-3-large":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-large")
    from langchain_fireworks import FireworksEmbeddings
    return FireworksEmbeddings(model=model_name)


# NOTE: via memory-agent
def split_model_and_provider(fully_specified_name: str) -> dict:
    """Initialize the configured chat model."""
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = None
        model = fully_specified_name
    return {"model": model, "provider": provider}


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        # case "cohere":
        #     from langchain_cohere import CohereEmbeddings

        #     return CohereEmbeddings(model=model)  # type: ignore
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")

def make_chat_model(name: str) -> BaseChatModel:
    """Connect to the configured chat model."""
    provider, model = name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider, temperature=0.0) # pyright: ignore[reportUndefinedVariable]

__all__ = ["ensure_configurable"]
