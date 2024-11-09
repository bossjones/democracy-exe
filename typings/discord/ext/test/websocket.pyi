"""
This type stub file was generated by pyright.
"""

import typing

import discord
import discord.gateway as gateway

from . import _types

"""
    Mock implementation of a ``discord.gateway.DiscordWebSocket``. Overwrites a Client's default websocket, allowing
    hooking of its methods to update the backend and provide callbacks.
"""
class FakeWebSocket(gateway.DiscordWebSocket):
    """
        A mock implementation of a ``DiscordWebSocket``. Instead of actually sending information to discord,
        it simply triggers calls to the ``dpytest`` backend, as well as triggering runner callbacks.
    """
    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        ...

    async def send(self, data: _types.JsonDict) -> None:
        ...

    async def change_presence(self, *, activity: typing.Optional[discord.BaseActivity] = ..., status: typing.Optional[str] = ..., since: float = ...) -> None:
        ...
