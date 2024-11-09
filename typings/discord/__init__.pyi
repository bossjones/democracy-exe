"""
This type stub file was generated by pyright.
"""

from typing import Literal, NamedTuple

from . import abc as abc
from . import app_commands as app_commands
from . import opus as opus
from . import ui as ui
from . import utils as utils
from .activity import *
from .appinfo import *
from .asset import *
from .audit_logs import *
from .automod import *
from .channel import *
from .client import *
from .colour import *
from .components import *
from .embeds import *
from .emoji import *
from .enums import *
from .errors import *
from .file import *
from .flags import *
from .guild import *
from .integrations import *
from .interactions import *
from .invite import *
from .member import *
from .mentions import *
from .message import *
from .object import *
from .partial_emoji import *
from .permissions import *
from .player import *
from .raw_models import *
from .reaction import *
from .role import *
from .scheduled_event import *
from .shard import *
from .stage_instance import *
from .sticker import *
from .team import *
from .template import *
from .threads import *
from .user import *
from .voice_client import *
from .webhook import *
from .welcome_screen import *
from .widget import *

"""
Discord API Wrapper
~~~~~~~~~~~~~~~~~~~

A basic wrapper for the Discord API.

:copyright: (c) 2015-present Rapptz
:license: MIT, see LICENSE for more details.

"""
__title__ = ...
__author__ = ...
__license__ = ...
__copyright__ = ...
__version__ = ...
__path__ = ...
class VersionInfo(NamedTuple):
    major: int
    minor: int
    micro: int
    releaselevel: Literal["alpha", "beta", "candidate", "final"]
    serial: int
    ...


version_info: VersionInfo = ...
