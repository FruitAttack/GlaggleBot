import os
import re
import asyncio
from datetime import timedelta

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from model_runtime import DiscordTextGenerator

load_dotenv()

TOKEN = os.environ["DISCORD_TOKEN"]

DEFAULT_PROMPT = "lol "
MAX_CONTEXT_MESSAGES = 50
MAX_PROMPT_CHARS = 500
MAX_OUTPUT_CHARS = 1900

generator = DiscordTextGenerator("discordModel.pt")

intents = discord.Intents.default()
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)

# Clean the messages for the model to use as context
def clean_discord_message(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"<@!?\d+>", "@user", text)
    text = re.sub(r"<#\d+>", "#channel", text)
    text = re.sub(r"<@&\d+>", "@role", text)

    return text

# Builds a prompt from recent/previous messages for the bot to continue from
async def get_recent_messages_prompt(channel: discord.abc.Messageable, recent_count: int = 25, minutes: int = 0) -> str:
    recent_count = max(1, min(recent_count, MAX_CONTEXT_MESSAGES))
    minutes = max(0, min(minutes, 60))

    after = None
    history_limit = recent_count + 10

    if minutes > 0:
        after = discord.utils.utcnow() - timedelta(minutes=minutes)
        history_limit = MAX_CONTEXT_MESSAGES + 10

    messages = []

    async for message in channel.history(
        limit=history_limit,
        after=after,
        oldest_first=False,
    ):
        if message.author.bot:
            continue

        cleaned = clean_discord_message(message.content)

        if cleaned:
            messages.append(cleaned)

        if minutes <= 0 and len(messages) >= recent_count:
            break

    messages.reverse()

    if not messages:
        return DEFAULT_PROMPT

    # Match your training format more closely.
    prompt = "\n<EOS>\n".join(messages)
    prompt = prompt[-MAX_PROMPT_CHARS:]

    # Add an EOS before generation so the model generates a next message
    return prompt + "\n<EOS>\n"

# have the model predict text
async def run_generation(prompt: str, max_chars: int, temperature: float) -> str:
    max_chars = max(1, min(max_chars, 1200))
    temperature = max(0.1, min(temperature, 2.0))

    text = await asyncio.to_thread(
        generator.generate,
        prompt,
        max_chars,
        temperature,
    )

    if not text:
        text = "I generated nothing. Try again."

    text = text[:MAX_OUTPUT_CHARS]

    return text

@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"Logged in as {bot.user}")


@bot.tree.command(
    name="generate_random",
    description="Generate a random message not relating to the current conversation",
)
@app_commands.describe(
    max_chars="How many new characters to generate",
    temperature="Lower = safer/repetitive, higher = weirder, try around 0.8 to start",
)
async def generaterandom(
    interaction: discord.Interaction,
    max_chars: int = 300,
    temperature: float = 0.8,
):
    await interaction.response.defer(thinking=True)

    text = await run_generation(
        prompt=DEFAULT_PROMPT,
        max_chars=max_chars,
        temperature=temperature,
    )

    await interaction.followup.send(
        text,
        allowed_mentions=discord.AllowedMentions.none(),
    )


@bot.tree.command(
    name="generate",
    description="Generate text using recent messages from this channel as context.",
)
@app_commands.describe(
    recent_count="How many recent messages to use if minutes is 0",
    minutes="Use messages from the last N minutes; 0 means use recent_count instead",
    max_chars="How many new characters to generate",
    temperature="Lower = safer/repetitive, higher = weirder, try around 0.8 to start",
)
async def generate(
    interaction: discord.Interaction,
    recent_count: int = 20,
    minutes: int = 0,
    max_chars: int = 128,
    temperature: float = 0.8,
):
    await interaction.response.defer(thinking=True)

    channel = interaction.channel

    if channel is None or not hasattr(channel, "history"):
        await interaction.followup.send(
            "I can't read message history in this channel.",
            ephemeral=True,
        )
        return

    prompt = await get_recent_messages_prompt(
        channel=channel,
        recent_count=recent_count,
        minutes=minutes,
    )

    text = await run_generation(
        prompt=prompt,
        max_chars=max_chars,
        temperature=temperature,
    )

    await interaction.followup.send(
        text,
        allowed_mentions=discord.AllowedMentions.none(),
    )


bot.run(TOKEN)

