# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
"""
A test Cog to only be used for testing koalabot.load_all_cogs

Commented using reStructuredText (reST)
"""
# Futures

# Built-in/Generic Imports
from __future__ import annotations

from typing import TYPE_CHECKING

import discord

from discord.ext import commands


# Own modules
if TYPE_CHECKING:
    from democracy_exe.chatbot.core.bot import DemocracyBot


class Greetings(commands.Cog):
    """
    A cog used for tests that greets the user
    """

    def __init__(self, bot):
        """
        Initialises class variables
        :param bot: The client of the bot being used
        """
        self.bot = bot
        self._last_member = None

    @commands.command()
    async def hello(self, ctx, *, member: discord.Member = None):
        """
        Says hello to the user
        :param ctx: context
        :param member: the member who sent the message
        """
        member = member or ctx.author
        if self._last_member is None or self._last_member.id != member.id:
            await ctx.send(f"Hello {member.name}~")
        else:
            await ctx.send(f"Hello {member.name}... This feels familiar.")
        self._last_member = member

    @commands.command()
    async def hi(self, ctx):
        """
        Says hi to the user
        :param ctx: The context of the message
        """
        await ctx.send(f"Hi {ctx.author}")


async def setup(bot: DemocracyBot) -> None:
    """
    Loads this cog into the selected bot
    :param bot: The client of the DemocracyBot
    """
    await bot.add_cog(Greetings(bot))
