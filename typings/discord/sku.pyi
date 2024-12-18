"""
This type stub file was generated by pyright.
"""

from typing import Optional, TYPE_CHECKING
from .flags import SKUFlags
from datetime import datetime
from .guild import Guild
from .state import ConnectionState
from .types.sku import Entitlement as EntitlementPayload, SKU as SKUPayload
from .user import User

"""
This type stub file was generated by pyright.
"""
if TYPE_CHECKING:
    ...
__all__ = ('SKU', 'Entitlement')
class SKU:
    """Represents a premium offering as a stock-keeping unit (SKU).

    .. versionadded:: 2.4

    Attributes
    -----------
    id: :class:`int`
        The SKU's ID.
    type: :class:`SKUType`
        The type of the SKU.
    application_id: :class:`int`
        The ID of the application that the SKU belongs to.
    name: :class:`str`
        The consumer-facing name of the premium offering.
    slug: :class:`str`
        A system-generated URL slug based on the SKU name.
    """
    __slots__ = ...
    def __init__(self, *, state: ConnectionState, data: SKUPayload) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def flags(self) -> SKUFlags:
        """:class:`SKUFlags`: Returns the flags of the SKU."""
        ...
    
    @property
    def created_at(self) -> datetime:
        """:class:`datetime.datetime`: Returns the sku's creation time in UTC."""
        ...
    


class Entitlement:
    """Represents an entitlement from user or guild which has been granted access to a premium offering.

    .. versionadded:: 2.4

    Attributes
    -----------
    id: :class:`int`
        The entitlement's ID.
    sku_id: :class:`int`
        The ID of the SKU that the entitlement belongs to.
    application_id: :class:`int`
        The ID of the application that the entitlement belongs to.
    user_id: Optional[:class:`int`]
        The ID of the user that is granted access to the entitlement.
    type: :class:`EntitlementType`
        The type of the entitlement.
    deleted: :class:`bool`
        Whether the entitlement has been deleted.
    starts_at: Optional[:class:`datetime.datetime`]
        A UTC start date which the entitlement is valid. Not present when using test entitlements.
    ends_at: Optional[:class:`datetime.datetime`]
        A UTC date which entitlement is no longer valid. Not present when using test entitlements.
    guild_id: Optional[:class:`int`]
        The ID of the guild that is granted access to the entitlement
    consumed: :class:`bool`
        For consumable items, whether the entitlement has been consumed.
    """
    __slots__ = ...
    def __init__(self, state: ConnectionState, data: EntitlementPayload) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    
    @property
    def user(self) -> Optional[User]:
        """Optional[:class:`User`]: The user that is granted access to the entitlement."""
        ...
    
    @property
    def guild(self) -> Optional[Guild]:
        """Optional[:class:`Guild`]: The guild that is granted access to the entitlement."""
        ...
    
    @property
    def created_at(self) -> datetime:
        """:class:`datetime.datetime`: Returns the entitlement's creation time in UTC."""
        ...
    
    def is_expired(self) -> bool:
        """:class:`bool`: Returns ``True`` if the entitlement is expired. Will be always False for test entitlements."""
        ...
    
    async def consume(self) -> None:
        """|coro|

        Marks a one-time purchase entitlement as consumed.

        Raises
        -------
        MissingApplicationID
            The application ID could not be found.
        NotFound
            The entitlement could not be found.
        HTTPException
            Consuming the entitlement failed.
        """
        ...
    
    async def delete(self) -> None:
        """|coro|

        Deletes the entitlement.

        Raises
        -------
        MissingApplicationID
            The application ID could not be found.
        NotFound
            The entitlement could not be found.
        HTTPException
            Deleting the entitlement failed.
        """
        ...
    


