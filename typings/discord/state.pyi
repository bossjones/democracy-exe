"""
This type stub file was generated by pyright.
"""

import asyncio
from typing import Any, Callable, Coroutine, Dict, Generic, List, Literal, Optional, Sequence, TYPE_CHECKING, Type, TypeVar, Union, overload
from .guild import Guild, GuildChannel
from .user import User
from .emoji import Emoji
from .partial_emoji import PartialEmoji
from .message import Message, MessageableChannel
from .channel import *
from .raw_models import *
from .member import Member
from .flags import Intents
from .ui.view import View
from .threads import Thread
from .sticker import GuildSticker
from ._types import ClientT
from .abc import PrivateChannel
from .http import HTTPClient
from .voice_client import VoiceProtocol
from .gateway import DiscordWebSocket
from .ui.item import Item
from .ui.dynamic import DynamicItem
from .types.automod import AutoModerationActionExecution, AutoModerationRule
from .types.channel import DMChannel as DMChannelPayload
from .types.user import PartialUser as PartialUserPayload, User as UserPayload
from .types.emoji import Emoji as EmojiPayload, PartialEmoji as PartialEmojiPayload
from .types.sticker import GuildSticker as GuildStickerPayload
from .types.message import Message as MessagePayload
from .types import gateway as gw
from .types.command import GuildApplicationCommandPermissions as GuildApplicationCommandPermissionsPayload

"""
This type stub file was generated by pyright.
"""
if TYPE_CHECKING:
    T = TypeVar('T')
    Channel = Union[GuildChannel, PrivateChannel, PartialMessageable]
class ChunkRequest:
    def __init__(self, guild_id: int, shard_id: int, loop: asyncio.AbstractEventLoop, resolver: Callable[[int], Any], *, cache: bool = ...) -> None:
        ...
    
    def add_members(self, members: List[Member]) -> None:
        ...
    
    async def wait(self) -> List[Member]:
        ...
    
    def get_future(self) -> asyncio.Future[List[Member]]:
        ...
    
    def done(self) -> None:
        ...
    


_log = ...
async def logging_coroutine(coroutine: Coroutine[Any, Any, T], *, info: str) -> Optional[T]:
    ...

class ConnectionState(Generic[ClientT]):
    if TYPE_CHECKING:
        _get_websocket: Callable[..., DiscordWebSocket]
        _get_client: Callable[..., ClientT]
        _parsers: Dict[str, Callable[[Dict[str, Any]], None]]
        ...
    def __init__(self, *, dispatch: Callable[..., Any], handlers: Dict[str, Callable[..., Any]], hooks: Dict[str, Callable[..., Coroutine[Any, Any, Any]]], http: HTTPClient, **options: Any) -> None:
        ...
    
    @property
    def cache_guild_expressions(self) -> bool:
        ...
    
    async def close(self) -> None:
        ...
    
    def clear(self, *, views: bool = ...) -> None:
        ...
    
    def process_chunk_requests(self, guild_id: int, nonce: Optional[str], members: List[Member], complete: bool) -> None:
        ...
    
    def clear_chunk_requests(self, shard_id: int | None) -> None:
        ...
    
    def call_handlers(self, key: str, *args: Any, **kwargs: Any) -> None:
        ...
    
    async def call_hooks(self, key: str, *args: Any, **kwargs: Any) -> None:
        ...
    
    @property
    def self_id(self) -> Optional[int]:
        ...
    
    @property
    def intents(self) -> Intents:
        ...
    
    @property
    def voice_clients(self) -> List[VoiceProtocol]:
        ...
    
    def store_user(self, data: Union[UserPayload, PartialUserPayload], *, cache: bool = ...) -> User:
        ...
    
    def store_user_no_intents(self, data: Union[UserPayload, PartialUserPayload], *, cache: bool = ...) -> User:
        ...
    
    def create_user(self, data: Union[UserPayload, PartialUserPayload]) -> User:
        ...
    
    def get_user(self, id: int) -> Optional[User]:
        ...
    
    def store_emoji(self, guild: Guild, data: EmojiPayload) -> Emoji:
        ...
    
    def store_sticker(self, guild: Guild, data: GuildStickerPayload) -> GuildSticker:
        ...
    
    def store_view(self, view: View, message_id: Optional[int] = ..., interaction_id: Optional[int] = ...) -> None:
        ...
    
    def prevent_view_updates_for(self, message_id: int) -> Optional[View]:
        ...
    
    def store_dynamic_items(self, *items: Type[DynamicItem[Item[Any]]]) -> None:
        ...
    
    def remove_dynamic_items(self, *items: Type[DynamicItem[Item[Any]]]) -> None:
        ...
    
    @property
    def persistent_views(self) -> Sequence[View]:
        ...
    
    @property
    def guilds(self) -> Sequence[Guild]:
        ...
    
    @property
    def emojis(self) -> Sequence[Emoji]:
        ...
    
    @property
    def stickers(self) -> Sequence[GuildSticker]:
        ...
    
    def get_emoji(self, emoji_id: Optional[int]) -> Optional[Emoji]:
        ...
    
    def get_sticker(self, sticker_id: Optional[int]) -> Optional[GuildSticker]:
        ...
    
    @property
    def private_channels(self) -> Sequence[PrivateChannel]:
        ...
    
    def add_dm_channel(self, data: DMChannelPayload) -> DMChannel:
        ...
    
    async def chunker(self, guild_id: int, query: str = ..., limit: int = ..., presences: bool = ..., *, nonce: Optional[str] = ...) -> None:
        ...
    
    async def query_members(self, guild: Guild, query: Optional[str], limit: int, user_ids: Optional[List[int]], cache: bool, presences: bool) -> List[Member]:
        ...
    
    def parse_ready(self, data: gw.ReadyEvent) -> None:
        ...
    
    def parse_resumed(self, data: gw.ResumedEvent) -> None:
        ...
    
    def parse_message_create(self, data: gw.MessageCreateEvent) -> None:
        ...
    
    def parse_message_delete(self, data: gw.MessageDeleteEvent) -> None:
        ...
    
    def parse_message_delete_bulk(self, data: gw.MessageDeleteBulkEvent) -> None:
        ...
    
    def parse_message_update(self, data: gw.MessageUpdateEvent) -> None:
        ...
    
    def parse_message_reaction_add(self, data: gw.MessageReactionAddEvent) -> None:
        ...
    
    def parse_message_reaction_remove_all(self, data: gw.MessageReactionRemoveAllEvent) -> None:
        ...
    
    def parse_message_reaction_remove(self, data: gw.MessageReactionRemoveEvent) -> None:
        ...
    
    def parse_message_reaction_remove_emoji(self, data: gw.MessageReactionRemoveEmojiEvent) -> None:
        ...
    
    def parse_interaction_create(self, data: gw.InteractionCreateEvent) -> None:
        ...
    
    def parse_presence_update(self, data: gw.PresenceUpdateEvent) -> None:
        ...
    
    def parse_user_update(self, data: gw.UserUpdateEvent) -> None:
        ...
    
    def parse_invite_create(self, data: gw.InviteCreateEvent) -> None:
        ...
    
    def parse_invite_delete(self, data: gw.InviteDeleteEvent) -> None:
        ...
    
    def parse_channel_delete(self, data: gw.ChannelDeleteEvent) -> None:
        ...
    
    def parse_channel_update(self, data: gw.ChannelUpdateEvent) -> None:
        ...
    
    def parse_channel_create(self, data: gw.ChannelCreateEvent) -> None:
        ...
    
    def parse_channel_pins_update(self, data: gw.ChannelPinsUpdateEvent) -> None:
        ...
    
    def parse_thread_create(self, data: gw.ThreadCreateEvent) -> None:
        ...
    
    def parse_thread_update(self, data: gw.ThreadUpdateEvent) -> None:
        ...
    
    def parse_thread_delete(self, data: gw.ThreadDeleteEvent) -> None:
        ...
    
    def parse_thread_list_sync(self, data: gw.ThreadListSyncEvent) -> None:
        ...
    
    def parse_thread_member_update(self, data: gw.ThreadMemberUpdate) -> None:
        ...
    
    def parse_thread_members_update(self, data: gw.ThreadMembersUpdate) -> None:
        ...
    
    def parse_guild_member_add(self, data: gw.GuildMemberAddEvent) -> None:
        ...
    
    def parse_guild_member_remove(self, data: gw.GuildMemberRemoveEvent) -> None:
        ...
    
    def parse_guild_member_update(self, data: gw.GuildMemberUpdateEvent) -> None:
        ...
    
    def parse_guild_emojis_update(self, data: gw.GuildEmojisUpdateEvent) -> None:
        ...
    
    def parse_guild_stickers_update(self, data: gw.GuildStickersUpdateEvent) -> None:
        ...
    
    def parse_guild_audit_log_entry_create(self, data: gw.GuildAuditLogEntryCreate) -> None:
        ...
    
    def parse_auto_moderation_rule_create(self, data: AutoModerationRule) -> None:
        ...
    
    def parse_auto_moderation_rule_update(self, data: AutoModerationRule) -> None:
        ...
    
    def parse_auto_moderation_rule_delete(self, data: AutoModerationRule) -> None:
        ...
    
    def parse_auto_moderation_action_execution(self, data: AutoModerationActionExecution) -> None:
        ...
    
    def is_guild_evicted(self, guild: Guild) -> bool:
        ...
    
    @overload
    async def chunk_guild(self, guild: Guild, *, wait: Literal[True] = ..., cache: Optional[bool] = ...) -> List[Member]:
        ...
    
    @overload
    async def chunk_guild(self, guild: Guild, *, wait: Literal[False] = ..., cache: Optional[bool] = ...) -> asyncio.Future[List[Member]]:
        ...
    
    async def chunk_guild(self, guild: Guild, *, wait: bool = ..., cache: Optional[bool] = ...) -> Union[List[Member], asyncio.Future[List[Member]]]:
        ...
    
    def parse_guild_create(self, data: gw.GuildCreateEvent) -> None:
        ...
    
    def parse_guild_update(self, data: gw.GuildUpdateEvent) -> None:
        ...
    
    def parse_guild_delete(self, data: gw.GuildDeleteEvent) -> None:
        ...
    
    def parse_guild_ban_add(self, data: gw.GuildBanAddEvent) -> None:
        ...
    
    def parse_guild_ban_remove(self, data: gw.GuildBanRemoveEvent) -> None:
        ...
    
    def parse_guild_role_create(self, data: gw.GuildRoleCreateEvent) -> None:
        ...
    
    def parse_guild_role_delete(self, data: gw.GuildRoleDeleteEvent) -> None:
        ...
    
    def parse_guild_role_update(self, data: gw.GuildRoleUpdateEvent) -> None:
        ...
    
    def parse_guild_members_chunk(self, data: gw.GuildMembersChunkEvent) -> None:
        ...
    
    def parse_guild_integrations_update(self, data: gw.GuildIntegrationsUpdateEvent) -> None:
        ...
    
    def parse_integration_create(self, data: gw.IntegrationCreateEvent) -> None:
        ...
    
    def parse_integration_update(self, data: gw.IntegrationUpdateEvent) -> None:
        ...
    
    def parse_integration_delete(self, data: gw.IntegrationDeleteEvent) -> None:
        ...
    
    def parse_webhooks_update(self, data: gw.WebhooksUpdateEvent) -> None:
        ...
    
    def parse_stage_instance_create(self, data: gw.StageInstanceCreateEvent) -> None:
        ...
    
    def parse_stage_instance_update(self, data: gw.StageInstanceUpdateEvent) -> None:
        ...
    
    def parse_stage_instance_delete(self, data: gw.StageInstanceDeleteEvent) -> None:
        ...
    
    def parse_guild_scheduled_event_create(self, data: gw.GuildScheduledEventCreateEvent) -> None:
        ...
    
    def parse_guild_scheduled_event_update(self, data: gw.GuildScheduledEventUpdateEvent) -> None:
        ...
    
    def parse_guild_scheduled_event_delete(self, data: gw.GuildScheduledEventDeleteEvent) -> None:
        ...
    
    def parse_guild_scheduled_event_user_add(self, data: gw.GuildScheduledEventUserAdd) -> None:
        ...
    
    def parse_guild_scheduled_event_user_remove(self, data: gw.GuildScheduledEventUserRemove) -> None:
        ...
    
    def parse_application_command_permissions_update(self, data: GuildApplicationCommandPermissionsPayload):
        ...
    
    def parse_voice_state_update(self, data: gw.VoiceStateUpdateEvent) -> None:
        ...
    
    def parse_voice_server_update(self, data: gw.VoiceServerUpdateEvent) -> None:
        ...
    
    def parse_typing_start(self, data: gw.TypingStartEvent) -> None:
        ...
    
    def parse_entitlement_create(self, data: gw.EntitlementCreateEvent) -> None:
        ...
    
    def parse_entitlement_update(self, data: gw.EntitlementUpdateEvent) -> None:
        ...
    
    def parse_entitlement_delete(self, data: gw.EntitlementDeleteEvent) -> None:
        ...
    
    def parse_message_poll_vote_add(self, data: gw.PollVoteActionEvent) -> None:
        ...
    
    def parse_message_poll_vote_remove(self, data: gw.PollVoteActionEvent) -> None:
        ...
    
    def get_reaction_emoji(self, data: PartialEmojiPayload) -> Union[Emoji, PartialEmoji, str]:
        ...
    
    def get_channel(self, id: Optional[int]) -> Optional[Union[Channel, Thread]]:
        ...
    
    def create_message(self, *, channel: MessageableChannel, data: MessagePayload) -> Message:
        ...
    


class AutoShardedConnectionState(ConnectionState[ClientT]):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    async def chunker(self, guild_id: int, query: str = ..., limit: int = ..., presences: bool = ..., *, shard_id: Optional[int] = ..., nonce: Optional[str] = ...) -> None:
        ...
    
    def parse_ready(self, data: gw.ReadyEvent) -> None:
        ...
    
    def parse_resumed(self, data: gw.ResumedEvent) -> None:
        ...
    


