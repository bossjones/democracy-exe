from __future__ import annotations

import os

from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Optional, Type, TypeVar

import prompts
import rich

from langchain_core.runnables import RunnableConfig, ensure_config
import structlog

logger = structlog.get_logger(__name__)
from rich.pretty import pprint


def _update_configurable_for_backwards_compatibility(
    configurable: dict[str, Any],
) -> dict[str, Any]:
    update = {}
    # if "k" in configurable:
    #     update["search_kwargs"] = {"k": configurable["k"]}

    if "model" in configurable:
        logger.error(f"Pre Configurable - model: {configurable['model']}")
        update["model"] = "openai/gpt-4o"
        logger.error(f"Post Configurable - model: {update['model']}")

    if "delay" in configurable:
        logger.error(f"Pre Configurable - delay: {configurable['delay']}")
        update["delay"] = 60
        logger.error(f"Post Configurable - delay: {update['delay']}")

    if "system_prompt" in configurable:
        logger.error(f"Pre Configurable - system_prompt: {configurable['system_prompt']}")
        update["system_prompt"] = prompts.MODEL_SYSTEM_MESSAGE
        logger.error(f"Post Configurable - system_prompt: {update['system_prompt']}")
    if update:
        return {**configurable, **update}

    return configurable

@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the chatbot."""
    user_id: str = "default-user"

    """The ID of the user to remember in the conversation."""
    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        # default="anthropic/claude-3-5-sonnet-20240620",
        default="openai/gpt-4o",
        metadata={
            "description": "The name of the language model to use for the agent. "
            "Should be in the form: provider/model-name."
        },
    )
    # for debounding memory creation
    delay: int = 60

    system_prompt: str = prompts.MODEL_SYSTEM_MESSAGE
    # trustcall_instruction: str = prompts.TRUSTCALL_INSTRUCTION

    @classmethod
    def from_runnable_config(
        cls: type[T], config: RunnableConfig | None | None = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        # logger.error(f"Config - config: {config}")
        # configurable = (
        #     config["configurable"] if config and "configurable" in config else {}
        # )
        configurable = _update_configurable_for_backwards_compatibility(configurable)
        rich.print("configurable:")
        pprint(configurable)
        # logger.error(f"Configurable - configurable after _update_configurable_for_backwards_compatibility: {configurable}")
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        # logger.error(f"Values - values: {values}")
        return cls(**{k: v for k, v in values.items() if v})

T = TypeVar("T", bound=Configuration)

if __name__ == "__main__":  # pragma: no cover
    import rich
    config = {}
    configurable = Configuration.from_runnable_config(config)
    rich.print(f"configurable: {configurable}")
