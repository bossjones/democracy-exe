"""
This type stub file was generated by pyright.
"""

import asyncio
import discord
import typing

"""
This type stub file was generated by pyright.
"""
def embed_eq(embed1: typing.Optional[discord.Embed], embed2: typing.Optional[discord.Embed]) -> bool:
    ...

def activity_eq(act1: typing.Optional[discord.Activity], act2: typing.Optional[discord.Activity]) -> bool:
    ...

def embed_proxy_eq(embed_proxy1, embed_proxy2):
    ...

class PeekableQueue(asyncio.Queue):
    """
        An extension of an asyncio queue with a peek message, so other code doesn't need to rely on unstable
        internal artifacts
    """
    def peek(self):
        """
            Peek the current last value in the queue, or raise an exception if there are no values

        :return: Last value in the queue, assuming there are any
        """
        ...
    


