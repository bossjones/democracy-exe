"""
This type stub file was generated by pyright.
"""

import datetime
from typing import Optional, TYPE_CHECKING, Union
from .asset import Asset
from .object import Object
from .mixins import Hashable
from typing_extensions import Self
from .types.invite import GatewayInvite as GatewayInvitePayload, Invite as InvitePayload, InviteGuild as InviteGuildPayload
from .types.channel import PartialChannel as InviteChannelPayload
from .state import ConnectionState
from .guild import Guild
from .abc import GuildChannel, Snowflake

"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
__all__ = ('PartialInviteChannel', 'PartialInviteGuild', 'Invite')
if TYPE_CHECKING:
    InviteGuildType = Union[Guild, 'PartialInviteGuild', Object]
    InviteChannelType = Union[GuildChannel, 'PartialInviteChannel', Object]
class PartialInviteChannel:
    """Represents a "partial" invite channel.

    This model will be given when the user is not part of the
    guild the :class:`Invite` resolves to.

    .. container:: operations

        .. describe:: x == y

            Checks if two partial channels are the same.

        .. describe:: x != y

            Checks if two partial channels are not the same.

        .. describe:: hash(x)

            Return the partial channel's hash.

        .. describe:: str(x)

            Returns the partial channel's name.

    Attributes
    -----------
    name: :class:`str`
        The partial channel's name.
    id: :class:`int`
        The partial channel's ID.
    type: :class:`ChannelType`
        The partial channel's type.
    """
    __slots__ = ...
    def __init__(self, data: InviteChannelPayload) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def mention(self) -> str:
        """:class:`str`: The string that allows you to mention the channel."""
        ...
    
    @property
    def created_at(self) -> datetime.datetime:
        """:class:`datetime.datetime`: Returns the channel's creation time in UTC."""
        ...
    


class PartialInviteGuild:
    """Represents a "partial" invite guild.

    This model will be given when the user is not part of the
    guild the :class:`Invite` resolves to.

    .. container:: operations

        .. describe:: x == y

            Checks if two partial guilds are the same.

        .. describe:: x != y

            Checks if two partial guilds are not the same.

        .. describe:: hash(x)

            Return the partial guild's hash.

        .. describe:: str(x)

            Returns the partial guild's name.

    Attributes
    -----------
    name: :class:`str`
        The partial guild's name.
    id: :class:`int`
        The partial guild's ID.
    verification_level: :class:`VerificationLevel`
        The partial guild's verification level.
    features: List[:class:`str`]
        A list of features the guild has. See :attr:`Guild.features` for more information.
    description: Optional[:class:`str`]
        The partial guild's description.
    nsfw_level: :class:`NSFWLevel`
        The partial guild's NSFW level.

        .. versionadded:: 2.0
    vanity_url_code: Optional[:class:`str`]
        The partial guild's vanity URL code, if available.

        .. versionadded:: 2.0
    premium_subscription_count: :class:`int`
        The number of "boosts" the partial guild currently has.

        .. versionadded:: 2.0
    """
    __slots__ = ...
    def __init__(self, state: ConnectionState, data: InviteGuildPayload, id: int) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def created_at(self) -> datetime.datetime:
        """:class:`datetime.datetime`: Returns the guild's creation time in UTC."""
        ...
    
    @property
    def vanity_url(self) -> Optional[str]:
        """Optional[:class:`str`]: The Discord vanity invite URL for this partial guild, if available.

        .. versionadded:: 2.0
        """
        ...
    
    @property
    def icon(self) -> Optional[Asset]:
        """Optional[:class:`Asset`]: Returns the guild's icon asset, if available."""
        ...
    
    @property
    def banner(self) -> Optional[Asset]:
        """Optional[:class:`Asset`]: Returns the guild's banner asset, if available."""
        ...
    
    @property
    def splash(self) -> Optional[Asset]:
        """Optional[:class:`Asset`]: Returns the guild's invite splash asset, if available."""
        ...
    


class Invite(Hashable):
    r"""Represents a Discord :class:`Guild` or :class:`abc.GuildChannel` invite.

    Depending on the way this object was created, some of the attributes can
    have a value of ``None``.

    .. container:: operations

        .. describe:: x == y

            Checks if two invites are equal.

        .. describe:: x != y

            Checks if two invites are not equal.

        .. describe:: hash(x)

            Returns the invite hash.

        .. describe:: str(x)

            Returns the invite URL.

    The following table illustrates what methods will obtain the attributes:

    +------------------------------------+--------------------------------------------------------------+
    |             Attribute              |                          Method                              |
    +====================================+==============================================================+
    | :attr:`max_age`                    | :meth:`abc.GuildChannel.invites`\, :meth:`Guild.invites`     |
    +------------------------------------+--------------------------------------------------------------+
    | :attr:`max_uses`                   | :meth:`abc.GuildChannel.invites`\, :meth:`Guild.invites`     |
    +------------------------------------+--------------------------------------------------------------+
    | :attr:`created_at`                 | :meth:`abc.GuildChannel.invites`\, :meth:`Guild.invites`     |
    +------------------------------------+--------------------------------------------------------------+
    | :attr:`temporary`                  | :meth:`abc.GuildChannel.invites`\, :meth:`Guild.invites`     |
    +------------------------------------+--------------------------------------------------------------+
    | :attr:`uses`                       | :meth:`abc.GuildChannel.invites`\, :meth:`Guild.invites`     |
    +------------------------------------+--------------------------------------------------------------+
    | :attr:`approximate_member_count`   | :meth:`Client.fetch_invite` with ``with_counts`` enabled     |
    +------------------------------------+--------------------------------------------------------------+
    | :attr:`approximate_presence_count` | :meth:`Client.fetch_invite` with ``with_counts`` enabled     |
    +------------------------------------+--------------------------------------------------------------+
    | :attr:`expires_at`                 | :meth:`Client.fetch_invite` with ``with_expiration`` enabled |
    +------------------------------------+--------------------------------------------------------------+

    If it's not in the table above then it is available by all methods.

    Attributes
    -----------
    type: :class:`InviteType`
        The type of the invite.

        .. versionadded: 2.4
    max_age: Optional[:class:`int`]
        How long before the invite expires in seconds.
        A value of ``0`` indicates that it doesn't expire.
    code: :class:`str`
        The URL fragment used for the invite.
    guild: Optional[Union[:class:`Guild`, :class:`Object`, :class:`PartialInviteGuild`]]
        The guild the invite is for. Can be ``None`` if it's from a group direct message.
    revoked: Optional[:class:`bool`]
        Indicates if the invite has been revoked.
    created_at: Optional[:class:`datetime.datetime`]
        An aware UTC datetime object denoting the time the invite was created.
    temporary: Optional[:class:`bool`]
        Indicates that the invite grants temporary membership.
        If ``True``, members who joined via this invite will be kicked upon disconnect.
    uses: Optional[:class:`int`]
        How many times the invite has been used.
    max_uses: Optional[:class:`int`]
        How many times the invite can be used.
        A value of ``0`` indicates that it has unlimited uses.
    inviter: Optional[:class:`User`]
        The user who created the invite.
    approximate_member_count: Optional[:class:`int`]
        The approximate number of members in the guild.
    approximate_presence_count: Optional[:class:`int`]
        The approximate number of members currently active in the guild.
        This includes idle, dnd, online, and invisible members. Offline members are excluded.
    expires_at: Optional[:class:`datetime.datetime`]
        The expiration date of the invite. If the value is ``None`` when received through
        :meth:`Client.fetch_invite` with ``with_expiration`` enabled, the invite will never expire.

        .. versionadded:: 2.0

    channel: Optional[Union[:class:`abc.GuildChannel`, :class:`Object`, :class:`PartialInviteChannel`]]
        The channel the invite is for.
    target_type: :class:`InviteTarget`
        The type of target for the voice channel invite.

        .. versionadded:: 2.0

    target_user: Optional[:class:`User`]
        The user whose stream to display for this invite, if any.

        .. versionadded:: 2.0

    target_application: Optional[:class:`PartialAppInfo`]
        The embedded application the invite targets, if any.

        .. versionadded:: 2.0
    scheduled_event: Optional[:class:`ScheduledEvent`]
        The scheduled event associated with this invite, if any.

        .. versionadded:: 2.0
    scheduled_event_id: Optional[:class:`int`]
        The ID of the scheduled event associated with this invite, if any.

        .. versionadded:: 2.0
    """
    __slots__ = ...
    BASE = ...
    def __init__(self, *, state: ConnectionState, data: InvitePayload, guild: Optional[Union[PartialInviteGuild, Guild]] = ..., channel: Optional[Union[PartialInviteChannel, GuildChannel]] = ...) -> None:
        ...
    
    @classmethod
    def from_incomplete(cls, *, state: ConnectionState, data: InvitePayload) -> Self:
        ...
    
    @classmethod
    def from_gateway(cls, *, state: ConnectionState, data: GatewayInvitePayload) -> Self:
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def __hash__(self) -> int:
        ...
    
    @property
    def id(self) -> str:
        """:class:`str`: Returns the proper code portion of the invite."""
        ...
    
    @property
    def url(self) -> str:
        """:class:`str`: A property that retrieves the invite URL."""
        ...
    
    def set_scheduled_event(self, scheduled_event: Snowflake, /) -> Self:
        """Sets the scheduled event for this invite.

        .. versionadded:: 2.0

        Parameters
        ----------
        scheduled_event: :class:`~discord.abc.Snowflake`
            The ID of the scheduled event.

        Returns
        --------
        :class:`Invite`
            The invite with the new scheduled event.
        """
        ...
    
    async def delete(self, *, reason: Optional[str] = ...) -> None:
        """|coro|

        Revokes the instant invite.

        You must have :attr:`~Permissions.manage_channels` to do this.

        Parameters
        -----------
        reason: Optional[:class:`str`]
            The reason for deleting this invite. Shows up on the audit log.

        Raises
        -------
        Forbidden
            You do not have permissions to revoke invites.
        NotFound
            The invite is invalid or expired.
        HTTPException
            Revoking the invite failed.
        """
        ...
    


