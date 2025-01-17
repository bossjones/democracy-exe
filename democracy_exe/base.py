"""Base classes and utilities for the Democracy AI system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


SEPARATOR_TOKEN = "<|endoftext|>"

###################################################################
# NOTE: on frozen dataclasses
###################################################################
# Frozen instances
# It is not possible to create truly immutable Python objects. However, by passing frozen=True to the @dataclass decorator you can emulate immutability. In that case, dataclasses will add __setattr__() and __delattr__() methods to the class. These methods will raise a FrozenInstanceError when invoked.

# There is a tiny performance penalty when using frozen=True: __init__() cannot use simple assignment to initialize fields, and must use object.__setattr__().
###################################################################


@dataclass(frozen=True)
class DemocracyMessage:
    """Represents a message in a Democracy conversation.

    Attributes:
        user: The user who sent the message.
        text: The text content of the message.
    """

    user: str
    text: str | None = None

    def render(self) -> str:
        """Renders the message as a string.

        Returns:
            The rendered message string.
        """
        result = self.user + ":"
        if self.text is not None:
            result += " " + self.text
        return result


@dataclass
class DemocracyConversation:
    """Represents a conversation in the Democracy system.

    Attributes:
        messages: The list of messages in the conversation.
    """

    messages: list[DemocracyMessage]

    def prepend(self, message: DemocracyMessage) -> DemocracyConversation:
        """Prepends a message to the conversation.

        Args:
            message: The message to prepend.

        Returns:
            The updated conversation.
        """
        self.messages.insert(0, message)
        return self

    def render(self) -> str:
        """Renders the conversation as a string.

        Returns:
            The rendered conversation string.
        """
        return f"\n{SEPARATOR_TOKEN}".join([message.render() for message in self.messages])


@dataclass(frozen=True)
class DemocracyConfig:
    """Configuration for a Democracy AI instance.

    Attributes:
        name: The name of the Democracy AI instance.
        instructions: The instructions for the Democracy AI.
        example_conversations: Example conversations for the Democracy AI.
    """

    name: str
    instructions: str
    example_conversations: list[DemocracyConversation]


@dataclass(frozen=True)
class DemocracyThreadConfig:
    """Configuration for a Democracy thread.

    Attributes:
        model: The name of the model to use.
        max_tokens: The maximum number of tokens to generate.
        temperature: The temperature for text generation.
    """

    model: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True)
class DemocracyPrompt:
    """Represents a prompt for the Democracy AI.

    Attributes:
        header: The header message of the prompt.
        examples: Example conversations for the prompt.
        convo: The current conversation for the prompt.
    """

    header: DemocracyMessage
    examples: list[DemocracyConversation]
    convo: DemocracyConversation

    def full_render(self, bot_name: str) -> list[dict]:
        """Renders the full prompt for the Democracy AI.

        Args:
            bot_name: The name of the bot.

        Returns:
            The rendered prompt as a list of message dictionaries.
        """
        messages = [
            {
                "role": "system",
                "content": self.render_system_prompt(),
            }
        ]
        for message in self.render_messages(bot_name):
            messages.append(message)
        return messages

    def render_system_prompt(self) -> str:
        """Renders the system prompt for the Democracy AI.

        Returns:
            The rendered system prompt string.
        """
        return f"\n{SEPARATOR_TOKEN}".join(
            [self.header.render()]
            + [DemocracyMessage("System", "Example conversations:").render()]
            + [conversation.render() for conversation in self.examples]
            + [DemocracyMessage("System", "Now, you will work with the actual current conversation.").render()]
        )

    def render_messages(self, bot_name: str) -> list[dict[str, str]]:
        """Renders the messages for the Democracy AI.

        This method converts the conversation messages into a format suitable for
        AI model input, distinguishing between user and assistant messages.

        Args:
            bot_name: The name of the bot, used to identify assistant messages.

        Returns:
            A list of dictionaries, each representing a message with 'role',
            'name', and 'content' keys. The 'role' is either 'user' or 'assistant'.

        Yields:
            dict[str, str]: A dictionary representing each message in the conversation.
        """
        for message in self.convo.messages:
            if bot_name not in message.user:
                yield {
                    "role": "user",
                    "name": message.user,
                    "content": message.text,
                }
            else:
                yield {
                    "role": "assistant",
                    "name": bot_name,
                    "content": message.text,
                }
