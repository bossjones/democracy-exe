# democracy_exe/chatbot/discord_bot.py
from __future__ import annotations

import discord

from discord.ext import commands

from democracy_exe.ai.graphs import AgentState, router_graph


class DemocracyBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

    async def on_ready(self):
        print(f"Logged in as {self.user}")

    async def on_message(self, message):
        if message.author == self.user:
            return

        state = AgentState(query=message.content, response="", current_agent="")
        result = router_graph.process(state)
        await message.channel.send(result["response"])

bot = DemocracyBot()
