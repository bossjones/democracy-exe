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


class Greeting(commands.Cog):
    @commands.Cog.listener()
    async def on_member_join(self, member):
        channel = member.guild.text_channels[0]
        if channel is not None:
            await channel.send(f"Welcome {member.mention}.")


async def setup(bot):
    await bot.add_cog(Greeting())
