"""
This type stub file was generated by pyright.
"""

from .snowflake import Snowflake
from typing import Literal, Optional, TypedDict
from typing_extensions import NotRequired

"""
This type stub file was generated by pyright.
"""
class AvatarDecorationData(TypedDict):
    asset: str
    sku_id: Snowflake
    ...


class PartialUser(TypedDict):
    id: Snowflake
    username: str
    discriminator: str
    avatar: Optional[str]
    global_name: Optional[str]
    avatar_decoration_data: NotRequired[AvatarDecorationData]
    ...


PremiumType = Literal[0, 1, 2, 3]
class User(PartialUser, total=False):
    bot: bool
    system: bool
    mfa_enabled: bool
    locale: str
    verified: bool
    email: Optional[str]
    flags: int
    premium_type: PremiumType
    public_flags: int
    ...


