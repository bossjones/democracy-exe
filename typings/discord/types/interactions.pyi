"""
This type stub file was generated by pyright.
"""

from typing import Dict, List, Literal, TYPE_CHECKING, TypedDict, Union
from typing_extensions import NotRequired
from .channel import ChannelTypeWithoutThread, GroupDMChannel, GuildChannel, InteractionDMChannel, ThreadMetadata
from .sku import Entitlement
from .threads import ThreadType
from .member import Member
from .message import Attachment, Message
from .role import Role
from .snowflake import Snowflake
from .user import User
from .guild import GuildFeature

"""
This type stub file was generated by pyright.
"""
if TYPE_CHECKING:
    ...
InteractionType = Literal[1, 2, 3, 4, 5]
InteractionContextType = Literal[0, 1, 2]
InteractionInstallationType = Literal[0, 1]
class _BasePartialChannel(TypedDict):
    id: Snowflake
    name: str
    permissions: str
    ...


class PartialChannel(_BasePartialChannel):
    type: ChannelTypeWithoutThread
    ...


class PartialThread(_BasePartialChannel):
    type: ThreadType
    thread_metadata: ThreadMetadata
    parent_id: Snowflake
    ...


class ResolvedData(TypedDict, total=False):
    users: Dict[str, User]
    members: Dict[str, Member]
    roles: Dict[str, Role]
    channels: Dict[str, Union[PartialChannel, PartialThread]]
    messages: Dict[str, Message]
    attachments: Dict[str, Attachment]
    ...


class PartialInteractionGuild(TypedDict):
    id: Snowflake
    locale: str
    features: List[GuildFeature]
    ...


class _BaseApplicationCommandInteractionDataOption(TypedDict):
    name: str
    ...


class _CommandGroupApplicationCommandInteractionDataOption(_BaseApplicationCommandInteractionDataOption):
    type: Literal[1, 2]
    options: List[ApplicationCommandInteractionDataOption]
    ...


class _BaseValueApplicationCommandInteractionDataOption(_BaseApplicationCommandInteractionDataOption, total=False):
    focused: bool
    ...


class _StringValueApplicationCommandInteractionDataOption(_BaseValueApplicationCommandInteractionDataOption):
    type: Literal[3]
    value: str
    ...


class _IntegerValueApplicationCommandInteractionDataOption(_BaseValueApplicationCommandInteractionDataOption):
    type: Literal[4]
    value: int
    ...


class _BooleanValueApplicationCommandInteractionDataOption(_BaseValueApplicationCommandInteractionDataOption):
    type: Literal[5]
    value: bool
    ...


class _SnowflakeValueApplicationCommandInteractionDataOption(_BaseValueApplicationCommandInteractionDataOption):
    type: Literal[6, 7, 8, 9, 11]
    value: Snowflake
    ...


class _NumberValueApplicationCommandInteractionDataOption(_BaseValueApplicationCommandInteractionDataOption):
    type: Literal[10]
    value: float
    ...


_ValueApplicationCommandInteractionDataOption = Union[_StringValueApplicationCommandInteractionDataOption, _IntegerValueApplicationCommandInteractionDataOption, _BooleanValueApplicationCommandInteractionDataOption, _SnowflakeValueApplicationCommandInteractionDataOption, _NumberValueApplicationCommandInteractionDataOption,]
ApplicationCommandInteractionDataOption = Union[_CommandGroupApplicationCommandInteractionDataOption, _ValueApplicationCommandInteractionDataOption,]
class _BaseApplicationCommandInteractionData(TypedDict):
    id: Snowflake
    name: str
    resolved: NotRequired[ResolvedData]
    guild_id: NotRequired[Snowflake]
    ...


class ChatInputApplicationCommandInteractionData(_BaseApplicationCommandInteractionData, total=False):
    type: Literal[1]
    options: List[ApplicationCommandInteractionDataOption]
    ...


class _BaseNonChatInputApplicationCommandInteractionData(_BaseApplicationCommandInteractionData):
    target_id: Snowflake
    ...


class UserApplicationCommandInteractionData(_BaseNonChatInputApplicationCommandInteractionData):
    type: Literal[2]
    ...


class MessageApplicationCommandInteractionData(_BaseNonChatInputApplicationCommandInteractionData):
    type: Literal[3]
    ...


ApplicationCommandInteractionData = Union[ChatInputApplicationCommandInteractionData, UserApplicationCommandInteractionData, MessageApplicationCommandInteractionData,]
class _BaseMessageComponentInteractionData(TypedDict):
    custom_id: str
    ...


class ButtonMessageComponentInteractionData(_BaseMessageComponentInteractionData):
    component_type: Literal[2]
    ...


class SelectMessageComponentInteractionData(_BaseMessageComponentInteractionData):
    component_type: Literal[3, 5, 6, 7, 8]
    values: List[str]
    resolved: NotRequired[ResolvedData]
    ...


MessageComponentInteractionData = Union[ButtonMessageComponentInteractionData, SelectMessageComponentInteractionData]
class ModalSubmitTextInputInteractionData(TypedDict):
    type: Literal[4]
    custom_id: str
    value: str
    ...


ModalSubmitComponentItemInteractionData = ModalSubmitTextInputInteractionData
class ModalSubmitActionRowInteractionData(TypedDict):
    type: Literal[1]
    components: List[ModalSubmitComponentItemInteractionData]
    ...


ModalSubmitComponentInteractionData = Union[ModalSubmitActionRowInteractionData, ModalSubmitComponentItemInteractionData]
class ModalSubmitInteractionData(TypedDict):
    custom_id: str
    components: List[ModalSubmitComponentInteractionData]
    ...


InteractionData = Union[ApplicationCommandInteractionData, MessageComponentInteractionData, ModalSubmitInteractionData,]
class _BaseInteraction(TypedDict):
    id: Snowflake
    application_id: Snowflake
    token: str
    version: Literal[1]
    guild_id: NotRequired[Snowflake]
    guild: NotRequired[PartialInteractionGuild]
    channel_id: NotRequired[Snowflake]
    channel: Union[GuildChannel, InteractionDMChannel, GroupDMChannel]
    app_permissions: NotRequired[str]
    locale: NotRequired[str]
    guild_locale: NotRequired[str]
    entitlement_sku_ids: NotRequired[List[Snowflake]]
    entitlements: NotRequired[List[Entitlement]]
    authorizing_integration_owners: Dict[Literal['0', '1'], Snowflake]
    context: NotRequired[InteractionContextType]
    ...


class PingInteraction(_BaseInteraction):
    type: Literal[1]
    ...


class ApplicationCommandInteraction(_BaseInteraction):
    type: Literal[2, 4]
    data: ApplicationCommandInteractionData
    ...


class MessageComponentInteraction(_BaseInteraction):
    type: Literal[3]
    data: MessageComponentInteractionData
    ...


class ModalSubmitInteraction(_BaseInteraction):
    type: Literal[5]
    data: ModalSubmitInteractionData
    ...


Interaction = Union[PingInteraction, ApplicationCommandInteraction, MessageComponentInteraction, ModalSubmitInteraction]
class MessageInteraction(TypedDict):
    id: Snowflake
    type: InteractionType
    name: str
    user: User
    member: NotRequired[Member]
    ...


class MessageInteractionMetadata(TypedDict):
    id: Snowflake
    type: InteractionType
    user: User
    authorizing_integration_owners: Dict[Literal['0', '1'], Snowflake]
    original_response_message_id: NotRequired[Snowflake]
    interacted_message_id: NotRequired[Snowflake]
    triggering_interaction_metadata: NotRequired[MessageInteractionMetadata]
    ...


