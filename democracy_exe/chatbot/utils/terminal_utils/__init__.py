"""Terminal utilities for the chatbot.

This package provides utilities for terminal-based bot operations including:
- Message handling and formatting
- Stream processing
- UI management
- Terminal I/O operations

The utilities are designed to work together to provide a clean and efficient
terminal interface while maintaining proper async operation and error handling.

Example:
    ```python
    from democracy_exe.chatbot.utils.terminal_utils import (
        MessageHandler,
        StreamHandler,
        UIManager
    )

    # Initialize handlers
    message_handler = MessageHandler()
    stream_handler = StreamHandler()
    ui_manager = UIManager()

    # Use in async context
    async with ui_manager as ui:
        await ui.display_welcome()
    ```
"""
from __future__ import annotations

from democracy_exe.chatbot.utils.terminal_utils.message_handler import MessageHandler
from democracy_exe.chatbot.utils.terminal_utils.stream_handler import StreamHandler
from democracy_exe.chatbot.utils.terminal_utils.ui_manager import UIManager


__all__ = [
    'MessageHandler',
    'StreamHandler',
    'UIManager',
]
