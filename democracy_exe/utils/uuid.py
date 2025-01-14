"""UUID utility functions"""

from __future__ import annotations

import uuid


def is_valid_uuid(val) -> bool:
    """Check UUID validity"""
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, TypeError):
        return False
