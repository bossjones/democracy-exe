# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
from __future__ import annotations

from discord.ext import commands
from discord.ext.commands import Cog, command


class Misc(Cog):
    # Silence the default on_error handler
    async def cog_command_error(self, ctx, error):
        pass

    @command()
    async def ping(self, ctx):
        await ctx.send("Pong !")


async def setup(bot):
    await bot.add_cog(Misc())
