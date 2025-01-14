"""
This type stub file was generated by pyright.
"""

from typing import List, Optional, TYPE_CHECKING
from .asset import Asset
from .flags import ApplicationFlags
from .permissions import Permissions
from .guild import Guild
from .types.appinfo import AppInfo as AppInfoPayload, InstallParams as InstallParamsPayload, PartialAppInfo as PartialAppInfoPayload
from .state import ConnectionState

"""
This type stub file was generated by pyright.
"""
if TYPE_CHECKING:
    ...
__all__ = ('AppInfo', 'PartialAppInfo', 'AppInstallParams')
class AppInfo:
    """Represents the application info for the bot provided by Discord.


    Attributes
    -------------
    id: :class:`int`
        The application ID.
    name: :class:`str`
        The application name.
    owner: :class:`User`
        The application owner.
    team: Optional[:class:`Team`]
        The application's team.

        .. versionadded:: 1.3

    description: :class:`str`
        The application description.
    bot_public: :class:`bool`
        Whether the bot can be invited by anyone or if it is locked
        to the application owner.
    bot_require_code_grant: :class:`bool`
        Whether the bot requires the completion of the full oauth2 code
        grant flow to join.
    rpc_origins: Optional[List[:class:`str`]]
        A list of RPC origin URLs, if RPC is enabled.

    verify_key: :class:`str`
        The hex encoded key for verification in interactions and the
        GameSDK's :ddocs:`GetTicket <game-sdk/applications#getticket>`.

        .. versionadded:: 1.3

    guild_id: Optional[:class:`int`]
        If this application is a game sold on Discord,
        this field will be the guild to which it has been linked to.

        .. versionadded:: 1.3

    primary_sku_id: Optional[:class:`int`]
        If this application is a game sold on Discord,
        this field will be the id of the "Game SKU" that is created,
        if it exists.

        .. versionadded:: 1.3

    slug: Optional[:class:`str`]
        If this application is a game sold on Discord,
        this field will be the URL slug that links to the store page.

        .. versionadded:: 1.3

    terms_of_service_url: Optional[:class:`str`]
        The application's terms of service URL, if set.

        .. versionadded:: 2.0

    privacy_policy_url: Optional[:class:`str`]
        The application's privacy policy URL, if set.

        .. versionadded:: 2.0

    tags: List[:class:`str`]
        The list of tags describing the functionality of the application.

        .. versionadded:: 2.0

    custom_install_url: List[:class:`str`]
        The custom authorization URL for the application, if enabled.

        .. versionadded:: 2.0

    install_params: Optional[:class:`AppInstallParams`]
        The settings for custom authorization URL of application, if enabled.

        .. versionadded:: 2.0
    role_connections_verification_url: Optional[:class:`str`]
        The application's connection verification URL which will render the application as
        a verification method in the guild's role verification configuration.

        .. versionadded:: 2.2
    interactions_endpoint_url: Optional[:class:`str`]
        The interactions endpoint url of the application to receive interactions over this endpoint rather than
        over the gateway, if configured.

        .. versionadded:: 2.4
    redirect_uris: List[:class:`str`]
        A list of authentication redirect URIs.

        .. versionadded:: 2.4
    approximate_guild_count: :class:`int`
        The approximate count of the guilds the bot was added to.

        .. versionadded:: 2.4
    """
    __slots__ = ...
    def __init__(self, state: ConnectionState, data: AppInfoPayload) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def icon(self) -> Optional[Asset]:
        """Optional[:class:`.Asset`]: Retrieves the application's icon asset, if any."""
        ...
    
    @property
    def cover_image(self) -> Optional[Asset]:
        """Optional[:class:`.Asset`]: Retrieves the cover image on a store embed, if any.

        This is only available if the application is a game sold on Discord.
        """
        ...
    
    @property
    def guild(self) -> Optional[Guild]:
        """Optional[:class:`Guild`]: If this application is a game sold on Discord,
        this field will be the guild to which it has been linked

        .. versionadded:: 1.3
        """
        ...
    
    @property
    def flags(self) -> ApplicationFlags:
        """:class:`ApplicationFlags`: The application's flags.

        .. versionadded:: 2.0
        """
        ...
    
    async def edit(self, *, reason: Optional[str] = ..., custom_install_url: Optional[str] = ..., description: Optional[str] = ..., role_connections_verification_url: Optional[str] = ..., install_params_scopes: Optional[List[str]] = ..., install_params_permissions: Optional[Permissions] = ..., flags: Optional[ApplicationFlags] = ..., icon: Optional[bytes] = ..., cover_image: Optional[bytes] = ..., interactions_endpoint_url: Optional[str] = ..., tags: Optional[List[str]] = ...) -> AppInfo:
        r"""|coro|

        Edits the application info.

        .. versionadded:: 2.4

        Parameters
        ----------
        custom_install_url: Optional[:class:`str`]
            The new custom authorization URL for the application. Can be ``None`` to remove the URL.
        description: Optional[:class:`str`]
            The new application description. Can be ``None`` to remove the description.
        role_connections_verification_url: Optional[:class:`str`]
            The new application's connection verification URL which will render the application
            as a verification method in the guild's role verification configuration. Can be ``None`` to remove the URL.
        install_params_scopes: Optional[List[:class:`str`]]
            The new list of :ddocs:`OAuth2 scopes <topics/oauth2#shared-resources-oauth2-scopes>` of
            the :attr:`~install_params`. Can be ``None`` to remove the scopes.
        install_params_permissions: Optional[:class:`Permissions`]
            The new permissions of the :attr:`~install_params`. Can be ``None`` to remove the permissions.
        flags: Optional[:class:`ApplicationFlags`]
            The new application's flags. Only limited intent flags (:attr:`~ApplicationFlags.gateway_presence_limited`,
            :attr:`~ApplicationFlags.gateway_guild_members_limited`, :attr:`~ApplicationFlags.gateway_message_content_limited`)
            can be edited. Can be ``None`` to remove the flags.

            .. warning::

                Editing the limited intent flags leads to the termination of the bot.

        icon: Optional[:class:`bytes`]
            The new application's icon as a :term:`py:bytes-like object`. Can be ``None`` to remove the icon.
        cover_image: Optional[:class:`bytes`]
            The new application's cover image as a :term:`py:bytes-like object` on a store embed.
            The cover image is only available if the application is a game sold on Discord.
            Can be ``None`` to remove the image.
        interactions_endpoint_url: Optional[:class:`str`]
            The new interactions endpoint url of the application to receive interactions over this endpoint rather than
            over the gateway. Can be ``None`` to remove the URL.
        tags: Optional[List[:class:`str`]]
            The new list of tags describing the functionality of the application. Can be ``None`` to remove the tags.
        reason: Optional[:class:`str`]
            The reason for editing the application. Shows up on the audit log.

        Raises
        -------
        HTTPException
            Editing the application failed
        ValueError
            The image format passed in to ``icon`` or ``cover_image`` is invalid. This is also raised
            when ``install_params_scopes`` and ``install_params_permissions`` are incompatible with each other.

        Returns
        -------
        :class:`AppInfo`
            The newly updated application info.
        """
        ...
    


class PartialAppInfo:
    """Represents a partial AppInfo given by :func:`~discord.abc.GuildChannel.create_invite`

    .. versionadded:: 2.0

    Attributes
    -------------
    id: :class:`int`
        The application ID.
    name: :class:`str`
        The application name.
    description: :class:`str`
        The application description.
    rpc_origins: Optional[List[:class:`str`]]
        A list of RPC origin URLs, if RPC is enabled.
    verify_key: :class:`str`
        The hex encoded key for verification in interactions and the
        GameSDK's :ddocs:`GetTicket <game-sdk/applications#getticket>`.
    terms_of_service_url: Optional[:class:`str`]
        The application's terms of service URL, if set.
    privacy_policy_url: Optional[:class:`str`]
        The application's privacy policy URL, if set.
    approximate_guild_count: :class:`int`
        The approximate count of the guilds the bot was added to.

        .. versionadded:: 2.3
    redirect_uris: List[:class:`str`]
        A list of authentication redirect URIs.

        .. versionadded:: 2.3
    interactions_endpoint_url: Optional[:class:`str`]
        The interactions endpoint url of the application to receive interactions over this endpoint rather than
        over the gateway, if configured.

        .. versionadded:: 2.3
    role_connections_verification_url: Optional[:class:`str`]
        The application's connection verification URL which will render the application as
        a verification method in the guild's role verification configuration.

        .. versionadded:: 2.3
    """
    __slots__ = ...
    def __init__(self, *, state: ConnectionState, data: PartialAppInfoPayload) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def icon(self) -> Optional[Asset]:
        """Optional[:class:`.Asset`]: Retrieves the application's icon asset, if any."""
        ...
    
    @property
    def cover_image(self) -> Optional[Asset]:
        """Optional[:class:`.Asset`]: Retrieves the cover image of the application's default rich presence.

        This is only available if the application is a game sold on Discord.

        .. versionadded:: 2.3
        """
        ...
    
    @property
    def flags(self) -> ApplicationFlags:
        """:class:`ApplicationFlags`: The application's flags.

        .. versionadded:: 2.0
        """
        ...
    


class AppInstallParams:
    """Represents the settings for custom authorization URL of an application.

    .. versionadded:: 2.0

    Attributes
    ----------
    scopes: List[:class:`str`]
        The list of :ddocs:`OAuth2 scopes <topics/oauth2#shared-resources-oauth2-scopes>`
        to add the application to a guild with.
    permissions: :class:`Permissions`
        The permissions to give to application in the guild.
    """
    __slots__ = ...
    def __init__(self, data: InstallParamsPayload) -> None:
        ...
    


