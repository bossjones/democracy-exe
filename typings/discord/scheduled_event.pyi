"""
This type stub file was generated by pyright.
"""

from datetime import datetime
from typing import AsyncIterator, Literal, Optional, TYPE_CHECKING, Union, overload
from .asset import Asset
from .enums import EntityType, EventStatus, PrivacyLevel
from .mixins import Hashable
from .types.scheduled_event import GuildScheduledEvent as BaseGuildScheduledEventPayload, GuildScheduledEventWithUserCount as GuildScheduledEventWithUserCountPayload
from .abc import Snowflake
from .guild import Guild
from .channel import StageChannel, VoiceChannel
from .state import ConnectionState
from .user import User

"""
This type stub file was generated by pyright.
"""
if TYPE_CHECKING:
    GuildScheduledEventPayload = Union[BaseGuildScheduledEventPayload, GuildScheduledEventWithUserCountPayload]
__all__ = ("ScheduledEvent", )
class ScheduledEvent(Hashable):
    """Represents a scheduled event in a guild.

    .. versionadded:: 2.0

    .. container:: operations

        .. describe:: x == y

            Checks if two scheduled events are equal.

        .. describe:: x != y

            Checks if two scheduled events are not equal.

        .. describe:: hash(x)

            Returns the scheduled event's hash.

    Attributes
    ----------
    id: :class:`int`
        The scheduled event's ID.
    name: :class:`str`
        The name of the scheduled event.
    description: Optional[:class:`str`]
        The description of the scheduled event.
    entity_type: :class:`EntityType`
        The type of entity this event is for.
    entity_id: Optional[:class:`int`]
        The ID of the entity this event is for if available.
    start_time: :class:`datetime.datetime`
        The time that the scheduled event will start in UTC.
    end_time: Optional[:class:`datetime.datetime`]
        The time that the scheduled event will end in UTC.
    privacy_level: :class:`PrivacyLevel`
        The privacy level of the scheduled event.
    status: :class:`EventStatus`
        The status of the scheduled event.
    user_count: :class:`int`
        The number of users subscribed to the scheduled event.
    creator: Optional[:class:`User`]
        The user that created the scheduled event.
    creator_id: Optional[:class:`int`]
        The ID of the user that created the scheduled event.

        .. versionadded:: 2.2
    location: Optional[:class:`str`]
        The location of the scheduled event.
    """
    __slots__ = ...
    def __init__(self, *, state: ConnectionState, data: GuildScheduledEventPayload) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def cover_image(self) -> Optional[Asset]:
        """Optional[:class:`Asset`]: The scheduled event's cover image."""
        ...
    
    @property
    def guild(self) -> Optional[Guild]:
        """Optional[:class:`Guild`]: The guild this scheduled event is in."""
        ...
    
    @property
    def channel(self) -> Optional[Union[VoiceChannel, StageChannel]]:
        """Optional[Union[:class:`VoiceChannel`, :class:`StageChannel`]]: The channel this scheduled event is in."""
        ...
    
    @property
    def url(self) -> str:
        """:class:`str`: The url for the scheduled event."""
        ...
    
    async def start(self, *, reason: Optional[str] = ...) -> ScheduledEvent:
        """|coro|

        Starts the scheduled event.

        Shorthand for:

        .. code-block:: python3

            await event.edit(status=EventStatus.active)

        Parameters
        -----------
        reason: Optional[:class:`str`]
            The reason for starting the scheduled event.

        Raises
        ------
        ValueError
            The scheduled event has already started or has ended.
        Forbidden
            You do not have the proper permissions to start the scheduled event.
        HTTPException
            The scheduled event could not be started.

        Returns
        -------
        :class:`ScheduledEvent`
            The scheduled event that was started.
        """
        ...
    
    async def end(self, *, reason: Optional[str] = ...) -> ScheduledEvent:
        """|coro|

        Ends the scheduled event.

        Shorthand for:

        .. code-block:: python3

            await event.edit(status=EventStatus.completed)

        Parameters
        -----------
        reason: Optional[:class:`str`]
            The reason for ending the scheduled event.

        Raises
        ------
        ValueError
            The scheduled event is not active or has already ended.
        Forbidden
            You do not have the proper permissions to end the scheduled event.
        HTTPException
            The scheduled event could not be ended.

        Returns
        -------
        :class:`ScheduledEvent`
            The scheduled event that was ended.
        """
        ...
    
    async def cancel(self, *, reason: Optional[str] = ...) -> ScheduledEvent:
        """|coro|

        Cancels the scheduled event.

        Shorthand for:

        .. code-block:: python3

            await event.edit(status=EventStatus.cancelled)

        Parameters
        -----------
        reason: Optional[:class:`str`]
            The reason for cancelling the scheduled event.

        Raises
        ------
        ValueError
            The scheduled event is already running.
        Forbidden
            You do not have the proper permissions to cancel the scheduled event.
        HTTPException
            The scheduled event could not be cancelled.

        Returns
        -------
        :class:`ScheduledEvent`
            The scheduled event that was cancelled.
        """
        ...
    
    @overload
    async def edit(self, *, name: str = ..., description: str = ..., start_time: datetime = ..., end_time: Optional[datetime] = ..., privacy_level: PrivacyLevel = ..., status: EventStatus = ..., image: bytes = ..., reason: Optional[str] = ...) -> ScheduledEvent:
        ...
    
    @overload
    async def edit(self, *, name: str = ..., description: str = ..., channel: Snowflake, start_time: datetime = ..., end_time: Optional[datetime] = ..., privacy_level: PrivacyLevel = ..., entity_type: Literal[EntityType.voice, EntityType.stage_instance], status: EventStatus = ..., image: bytes = ..., reason: Optional[str] = ...) -> ScheduledEvent:
        ...
    
    @overload
    async def edit(self, *, name: str = ..., description: str = ..., start_time: datetime = ..., end_time: datetime = ..., privacy_level: PrivacyLevel = ..., entity_type: Literal[EntityType.external], status: EventStatus = ..., image: bytes = ..., location: str, reason: Optional[str] = ...) -> ScheduledEvent:
        ...
    
    @overload
    async def edit(self, *, name: str = ..., description: str = ..., channel: Union[VoiceChannel, StageChannel], start_time: datetime = ..., end_time: Optional[datetime] = ..., privacy_level: PrivacyLevel = ..., status: EventStatus = ..., image: bytes = ..., reason: Optional[str] = ...) -> ScheduledEvent:
        ...
    
    @overload
    async def edit(self, *, name: str = ..., description: str = ..., start_time: datetime = ..., end_time: datetime = ..., privacy_level: PrivacyLevel = ..., status: EventStatus = ..., image: bytes = ..., location: str, reason: Optional[str] = ...) -> ScheduledEvent:
        ...
    
    async def edit(self, *, name: str = ..., description: str = ..., channel: Optional[Snowflake] = ..., start_time: datetime = ..., end_time: Optional[datetime] = ..., privacy_level: PrivacyLevel = ..., entity_type: EntityType = ..., status: EventStatus = ..., image: bytes = ..., location: str = ..., reason: Optional[str] = ...) -> ScheduledEvent:
        r"""|coro|

        Edits the scheduled event.

        You must have :attr:`~Permissions.manage_events` to do this.

        Parameters
        -----------
        name: :class:`str`
            The name of the scheduled event.
        description: :class:`str`
            The description of the scheduled event.
        channel: Optional[:class:`~discord.abc.Snowflake`]
            The channel to put the scheduled event in. If the channel is
            a :class:`StageInstance` or :class:`VoiceChannel` then
            it automatically sets the entity type.

            Required if the entity type is either :attr:`EntityType.voice` or
            :attr:`EntityType.stage_instance`.
        start_time: :class:`datetime.datetime`
            The time that the scheduled event will start. This must be a timezone-aware
            datetime object. Consider using :func:`utils.utcnow`.
        end_time: Optional[:class:`datetime.datetime`]
            The time that the scheduled event will end. This must be a timezone-aware
            datetime object. Consider using :func:`utils.utcnow`.

            If the entity type is either :attr:`EntityType.voice` or
            :attr:`EntityType.stage_instance`, the end_time can be cleared by
            passing ``None``.

            Required if the entity type is :attr:`EntityType.external`.
        privacy_level: :class:`PrivacyLevel`
            The privacy level of the scheduled event.
        entity_type: :class:`EntityType`
            The new entity type. If the channel is a :class:`StageInstance`
            or :class:`VoiceChannel` then this is automatically set to the
            appropriate entity type.
        status: :class:`EventStatus`
            The new status of the scheduled event.
        image: Optional[:class:`bytes`]
            The new image of the scheduled event or ``None`` to remove the image.
        location: :class:`str`
            The new location of the scheduled event.

            Required if the entity type is :attr:`EntityType.external`.
        reason: Optional[:class:`str`]
            The reason for editing the scheduled event. Shows up on the audit log.

        Raises
        -------
        TypeError
            ``image`` was not a :term:`py:bytes-like object`, or ``privacy_level``
            was not a :class:`PrivacyLevel`, or ``entity_type`` was not an
            :class:`EntityType`, ``status`` was not an :class:`EventStatus`, or
            an argument was provided that was incompatible with the scheduled event's
            entity type.
        ValueError
            ``start_time`` or ``end_time`` was not a timezone-aware datetime object.
        Forbidden
            You do not have permissions to edit the scheduled event.
        HTTPException
            Editing the scheduled event failed.

        Returns
        --------
        :class:`ScheduledEvent`
            The edited scheduled event.
        """
        ...
    
    async def delete(self, *, reason: Optional[str] = ...) -> None:
        """|coro|

        Deletes the scheduled event.

        You must have :attr:`~Permissions.manage_events` to do this.

        Parameters
        -----------
        reason: Optional[:class:`str`]
            The reason for deleting the scheduled event. Shows up on the audit log.

        Raises
        ------
        Forbidden
            You do not have permissions to delete the scheduled event.
        HTTPException
            Deleting the scheduled event failed.
        """
        ...
    
    async def users(self, *, limit: Optional[int] = ..., before: Optional[Snowflake] = ..., after: Optional[Snowflake] = ..., oldest_first: bool = ...) -> AsyncIterator[User]:
        """|coro|

        Retrieves all :class:`User` that are subscribed to this event.

        This requires :attr:`Intents.members` to get information about members
        other than yourself.

        Raises
        -------
        HTTPException
            Retrieving the members failed.

        Returns
        --------
        List[:class:`User`]
            All subscribed users of this event.
        """
        ...
    


