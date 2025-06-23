# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=no-value-for-parameter
# pylint: disable=possibly-used-before-assignment
# pyright: reportAttributeAccessIssue=false
# pyright: reportInvalidTypeForm=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUndefinedVariable=false
from __future__ import annotations

from discord.ext.commands import Cog, command


class Echo(Cog):
    # Silence the default on_error handler
    async def cog_command_error(self, ctx, error):
        pass

    @command()
    async def echo(self, ctx, *, text: str):
        await ctx.send(text)


async def setup(bot):
    await bot.add_cog(Echo())
