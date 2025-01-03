"""
This type stub file was generated by pyright.
"""

import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, TYPE_CHECKING, Union, overload
from .enums import AutoModRuleActionType, AutoModRuleEventType, AutoModRuleTriggerType
from .flags import AutoModPresets
from .utils import cached_slot_property
from typing_extensions import Self
from .abc import GuildChannel, Snowflake
from .threads import Thread
from .guild import Guild
from .member import Member
from .state import ConnectionState
from .types.automod import AutoModerationAction as AutoModerationActionPayload, AutoModerationActionExecution as AutoModerationActionExecutionPayload, AutoModerationRule as AutoModerationRulePayload, AutoModerationTriggerMetadata as AutoModerationTriggerMetadataPayload
from .role import Role

"""
This type stub file was generated by pyright.
"""
if TYPE_CHECKING:
    ...
__all__ = ('AutoModRuleAction', 'AutoModTrigger', 'AutoModRule', 'AutoModAction')
class AutoModRuleAction:
    """Represents an auto moderation's rule action.

    .. note::
        Only one of ``channel_id``, ``duration``, or ``custom_message`` can be used.

    .. versionadded:: 2.0

    Attributes
    -----------
    type: :class:`AutoModRuleActionType`
        The type of action to take.
        Defaults to :attr:`~AutoModRuleActionType.block_message`.
    channel_id: Optional[:class:`int`]
        The ID of the channel or thread to send the alert message to, if any.
        Passing this sets :attr:`type` to :attr:`~AutoModRuleActionType.send_alert_message`.
    duration: Optional[:class:`datetime.timedelta`]
        The duration of the timeout to apply, if any.
        Has a maximum of 28 days.
        Passing this sets :attr:`type` to :attr:`~AutoModRuleActionType.timeout`.
    custom_message: Optional[:class:`str`]
        A custom message which will be shown to a user when their message is blocked.
        Passing this sets :attr:`type` to :attr:`~AutoModRuleActionType.block_message`.

        .. versionadded:: 2.2
    """
    __slots__ = ...
    @overload
    def __init__(self, *, channel_id: int = ...) -> None:
        ...
    
    @overload
    def __init__(self, *, type: Literal[AutoModRuleActionType.send_alert_message], channel_id: int = ...) -> None:
        ...
    
    @overload
    def __init__(self, *, duration: datetime.timedelta = ...) -> None:
        ...
    
    @overload
    def __init__(self, *, type: Literal[AutoModRuleActionType.timeout], duration: datetime.timedelta = ...) -> None:
        ...
    
    @overload
    def __init__(self, *, custom_message: str = ...) -> None:
        ...
    
    @overload
    def __init__(self, *, type: Literal[AutoModRuleActionType.block_message]) -> None:
        ...
    
    @overload
    def __init__(self, *, type: Literal[AutoModRuleActionType.block_message], custom_message: Optional[str] = ...) -> None:
        ...
    
    @overload
    def __init__(self, *, type: Optional[AutoModRuleActionType] = ..., channel_id: Optional[int] = ..., duration: Optional[datetime.timedelta] = ..., custom_message: Optional[str] = ...) -> None:
        ...
    
    def __init__(self, *, type: Optional[AutoModRuleActionType] = ..., channel_id: Optional[int] = ..., duration: Optional[datetime.timedelta] = ..., custom_message: Optional[str] = ...) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @classmethod
    def from_data(cls, data: AutoModerationActionPayload) -> Self:
        ...
    
    def to_dict(self) -> Dict[str, Any]:
        ...
    


class AutoModTrigger:
    r"""Represents a trigger for an auto moderation rule.

    The following table illustrates relevant attributes for each :class:`AutoModRuleTriggerType`:

    +-----------------------------------------------+------------------------------------------------+
    |                    Type                       |                   Attributes                   |
    +===============================================+================================================+
    | :attr:`AutoModRuleTriggerType.keyword`        | :attr:`keyword_filter`, :attr:`regex_patterns`,|
    |                                               | :attr:`allow_list`                             |
    +-----------------------------------------------+------------------------------------------------+
    | :attr:`AutoModRuleTriggerType.spam`           |                                                |
    +-----------------------------------------------+------------------------------------------------+
    | :attr:`AutoModRuleTriggerType.keyword_preset` | :attr:`presets`\, :attr:`allow_list`           |
    +-----------------------------------------------+------------------------------------------------+
    | :attr:`AutoModRuleTriggerType.mention_spam`   | :attr:`mention_limit`,                         |
    |                                               | :attr:`mention_raid_protection`                |
    +-----------------------------------------------+------------------------------------------------+
    | :attr:`AutoModRuleTriggerType.member_profile` | :attr:`keyword_filter`, :attr:`regex_patterns`,|
    |                                               | :attr:`allow_list`                             |
    +-----------------------------------------------+------------------------------------------------+

    .. versionadded:: 2.0

    Attributes
    -----------
    type: :class:`AutoModRuleTriggerType`
        The type of trigger.
    keyword_filter: List[:class:`str`]
        The list of strings that will trigger the filter.
        Maximum of 1000. Keywords can only be up to 60 characters in length.

        This could be combined with :attr:`regex_patterns`.
    regex_patterns: List[:class:`str`]
        The regex pattern that will trigger the filter. The syntax is based off of
        `Rust's regex syntax <https://docs.rs/regex/latest/regex/#syntax>`_.
        Maximum of 10. Regex strings can only be up to 260 characters in length.

        This could be combined with :attr:`keyword_filter` and/or :attr:`allow_list`

        .. versionadded:: 2.1
    presets: :class:`AutoModPresets`
        The presets used with the preset keyword filter.
    allow_list: List[:class:`str`]
        The list of words that are exempt from the commonly flagged words. Maximum of 100.
        Keywords can only be up to 60 characters in length.
    mention_limit: :class:`int`
        The total number of user and role mentions a message can contain.
        Has a maximum of 50.
    mention_raid_protection: :class:`bool`
        Whether mention raid protection is enabled or not.

        .. versionadded:: 2.4
    """
    __slots__ = ...
    def __init__(self, *, type: Optional[AutoModRuleTriggerType] = ..., keyword_filter: Optional[List[str]] = ..., presets: Optional[AutoModPresets] = ..., allow_list: Optional[List[str]] = ..., mention_limit: Optional[int] = ..., regex_patterns: Optional[List[str]] = ..., mention_raid_protection: Optional[bool] = ...) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @classmethod
    def from_data(cls, type: int, data: Optional[AutoModerationTriggerMetadataPayload]) -> Self:
        ...
    
    def to_metadata_dict(self) -> Optional[Dict[str, Any]]:
        ...
    


class AutoModRule:
    """Represents an auto moderation rule.

    .. versionadded:: 2.0

    Attributes
    -----------
    id: :class:`int`
        The ID of the rule.
    guild: :class:`Guild`
        The guild the rule is for.
    name: :class:`str`
        The name of the rule.
    creator_id: :class:`int`
        The ID of the user that created the rule.
    trigger: :class:`AutoModTrigger`
        The rule's trigger.
    enabled: :class:`bool`
        Whether the rule is enabled.
    exempt_role_ids: Set[:class:`int`]
        The IDs of the roles that are exempt from the rule.
    exempt_channel_ids: Set[:class:`int`]
        The IDs of the channels that are exempt from the rule.
    event_type: :class:`AutoModRuleEventType`
        The type of event that will trigger the the rule.
    """
    __slots__ = ...
    def __init__(self, *, data: AutoModerationRulePayload, guild: Guild, state: ConnectionState) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def to_dict(self) -> AutoModerationRulePayload:
        ...
    
    @property
    def creator(self) -> Optional[Member]:
        """Optional[:class:`Member`]: The member that created this rule."""
        ...
    
    @cached_slot_property('_cs_exempt_roles')
    def exempt_roles(self) -> List[Role]:
        """List[:class:`Role`]: The roles that are exempt from this rule."""
        ...
    
    @cached_slot_property('_cs_exempt_channels')
    def exempt_channels(self) -> List[Union[GuildChannel, Thread]]:
        """List[Union[:class:`abc.GuildChannel`, :class:`Thread`]]: The channels that are exempt from this rule."""
        ...
    
    @cached_slot_property('_cs_actions')
    def actions(self) -> List[AutoModRuleAction]:
        """List[:class:`AutoModRuleAction`]: The actions that are taken when this rule is triggered."""
        ...
    
    def is_exempt(self, obj: Snowflake, /) -> bool:
        """Check if an object is exempt from the automod rule.

        Parameters
        -----------
        obj: :class:`abc.Snowflake`
            The role, channel, or thread to check.

        Returns
        --------
        :class:`bool`
            Whether the object is exempt from the automod rule.
        """
        ...
    
    async def edit(self, *, name: str = ..., event_type: AutoModRuleEventType = ..., actions: List[AutoModRuleAction] = ..., trigger: AutoModTrigger = ..., enabled: bool = ..., exempt_roles: Sequence[Snowflake] = ..., exempt_channels: Sequence[Snowflake] = ..., reason: str = ...) -> Self:
        """|coro|

        Edits this auto moderation rule.

        You must have :attr:`Permissions.manage_guild` to edit rules.

        Parameters
        -----------
        name: :class:`str`
            The new name to change to.
        event_type: :class:`AutoModRuleEventType`
            The new event type to change to.
        actions: List[:class:`AutoModRuleAction`]
            The new rule actions to update.
        trigger: :class:`AutoModTrigger`
            The new trigger to update.
            You can only change the trigger metadata, not the type.
        enabled: :class:`bool`
            Whether the rule should be enabled or not.
        exempt_roles: Sequence[:class:`abc.Snowflake`]
            The new roles to exempt from the rule.
        exempt_channels: Sequence[:class:`abc.Snowflake`]
            The new channels to exempt from the rule.
        reason: :class:`str`
            The reason for updating this rule. Shows up on the audit log.

        Raises
        -------
        Forbidden
            You do not have permission to edit this rule.
        HTTPException
            Editing the rule failed.

        Returns
        --------
        :class:`AutoModRule`
            The updated auto moderation rule.
        """
        ...
    
    async def delete(self, *, reason: str = ...) -> None:
        """|coro|

        Deletes the auto moderation rule.

        You must have :attr:`Permissions.manage_guild` to delete rules.

        Parameters
        -----------
        reason: :class:`str`
            The reason for deleting this rule. Shows up on the audit log.

        Raises
        -------
        Forbidden
            You do not have permissions to delete the rule.
        HTTPException
            Deleting the rule failed.
        """
        ...
    


class AutoModAction:
    """Represents an action that was taken as the result of a moderation rule.

    .. versionadded:: 2.0

    Attributes
    -----------
    action: :class:`AutoModRuleAction`
        The action that was taken.
    message_id: Optional[:class:`int`]
        The message ID that triggered the action. This is only available if the
        action is done on an edited message.
    rule_id: :class:`int`
        The ID of the rule that was triggered.
    rule_trigger_type: :class:`AutoModRuleTriggerType`
        The trigger type of the rule that was triggered.
    guild_id: :class:`int`
        The ID of the guild where the rule was triggered.
    user_id: :class:`int`
        The ID of the user that triggered the rule.
    channel_id: :class:`int`
        The ID of the channel where the rule was triggered.
    alert_system_message_id: Optional[:class:`int`]
        The ID of the system message that was sent to the predefined alert channel.
    content: :class:`str`
        The content of the message that triggered the rule.
        Requires the :attr:`Intents.message_content` or it will always return an empty string.
    matched_keyword: Optional[:class:`str`]
        The matched keyword from the triggering message.
    matched_content: Optional[:class:`str`]
        The matched content from the triggering message.
        Requires the :attr:`Intents.message_content` or it will always return ``None``.
    """
    __slots__ = ...
    def __init__(self, *, data: AutoModerationActionExecutionPayload, state: ConnectionState) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def guild(self) -> Guild:
        """:class:`Guild`: The guild this action was taken in."""
        ...
    
    @property
    def channel(self) -> Optional[Union[GuildChannel, Thread]]:
        """Optional[Union[:class:`abc.GuildChannel`, :class:`Thread`]]: The channel this action was taken in."""
        ...
    
    @property
    def member(self) -> Optional[Member]:
        """Optional[:class:`Member`]: The member this action was taken against /who triggered this rule."""
        ...
    
    async def fetch_rule(self) -> AutoModRule:
        """|coro|

        Fetch the rule whose action was taken.

        You must have :attr:`Permissions.manage_guild` to do this.

        Raises
        -------
        Forbidden
            You do not have permissions to view the rule.
        HTTPException
            Fetching the rule failed.

        Returns
        --------
        :class:`AutoModRule`
            The rule that was executed.
        """
        ...
    


