"""Constants for the application."""
from __future__ import annotations

import enum
import os

from pathlib import Path

from democracy_exe.aio_settings import aiosettings


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

# LLM settings
MAX_TOKENS = aiosettings.llm_max_tokens
MAX_RETRIES = aiosettings.llm_max_retries
RETRY_DELAY = aiosettings.llm_retry_delay
RETRY_MULTIPLIER = aiosettings.llm_retry_multiplier
RETRY_MAX_DELAY = aiosettings.llm_retry_max_delay

# Resource limits
MAX_MEMORY_MB = aiosettings.max_memory_mb
MAX_TASKS = aiosettings.max_tasks
MAX_RESPONSE_SIZE_MB = aiosettings.max_response_size_mb
MAX_BUFFER_SIZE_KB = aiosettings.max_buffer_size_kb
TASK_TIMEOUT_SECONDS = aiosettings.task_timeout_seconds

# File paths
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"

# Discord settings
PREFIX = aiosettings.prefix
CHANNEL_ID = "1240294186201124929"
CHAT_HISTORY_BUFFER = getattr(aiosettings, "chat_history_buffer", 10)

# Discord upload limits
MAX_BYTES_UPLOAD_DISCORD = 50000000
MAX_FILE_UPLOAD_IMAGES_IMGUR = 20000000
MAX_FILE_UPLOAD_VIDEO_IMGUR = 200000000
MAX_RUNTIME_VIDEO_IMGUR = 20  # seconds

# Twitter download commands
DL_SAFE_TWITTER_COMMAND = "gallery-dl --no-mtime {dl_uri}"
DL_TWITTER_THREAD_COMMAND = "gallery-dl --no-mtime --filter thread {dl_uri}"
DL_TWITTER_CARD_COMMAND = "gallery-dl --no-mtime --filter card {dl_uri}"

# Bot settings
SECONDS_DELAY_RECEIVING_MSG = 3  # give a delay for the bot to respond so it can catch multiple messages
MAX_THREAD_MESSAGES = 200
ACTIVATE_THREAD_PREFX = "💬✅"
INACTIVATE_THREAD_PREFIX = "💬❌"
MAX_CHARS_PER_REPLY_MSG = 1500  # discord has a 2k limit, we just break message into 1.5k

# Time constants
DAY_IN_SECONDS = 24 * 3600

# Numeric constants
ONE_MILLION = 1000000
FIVE_HUNDRED_THOUSAND = 500000
ONE_HUNDRED_THOUSAND = 100000
FIFTY_THOUSAND = 50000
THIRTY_THOUSAND = 30000
TWENTY_THOUSAND = 20000
TEN_THOUSAND = 10000
FIVE_THOUSAND = 5000

# Vector store settings
class SupportedVectorStores(str, enum.Enum):
    """Supported vector store types."""

    chroma = "chroma"
    milvus = "milvus"
    pgvector = "pgvector"
    pinecone = "pinecone"
    qdrant = "qdrant"
    weaviate = "weaviate"


class SupportedEmbeddings(str, enum.Enum):
    """Supported embedding types."""

    openai = "OpenAI"
    cohere = "Cohere"


# FIXME: THIS IS NOT CONCURRENCY SAFE
# Ensure directories exist
for directory in [CONFIG_DIR, DATA_DIR, LOGS_DIR, CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
