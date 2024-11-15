"""Factories for the sandbox agent."""

# Import factories from other modules as needed
from __future__ import annotations

import dataclasses

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# from democracy_exe.ai.chat_models import ChatModelFactory
# from democracy_exe.ai.document_loaders import DocumentLoaderFactory
# from democracy_exe.ai.embedding_models import EmbeddingModelFactory
# from democracy_exe.ai.evaluators import EvaluatorFactory
# from democracy_exe.ai.key_value_stores import KeyValueStoreFactory
# from democracy_exe.ai.memory import MemoryFactory
# from democracy_exe.ai.retrievers import RetrieverFactory
# from democracy_exe.ai.text_splitters import TextSplitterFactory
# from democracy_exe.ai.tools import ToolFactory
# from democracy_exe.ai.vector_stores import VectorStoreFactory


READ_ONLY = "read_only"
COERCE_TO = "coerce_to"


@dataclass
class SerializerFactory:
    def as_dict(self) -> dict:
        d: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if READ_ONLY not in f.metadata:
                if COERCE_TO in f.metadata:
                    value = f.metadata[COERCE_TO](value)
                d[f.name] = value
        return d
