# pyright: reportMissingTypeStubs=false
# pylint: disable=no-member
# pylint: disable=no-value-for-parameter
# pyright: reportAttributeAccessIssue=false
# pyright: reportImportCycles=false

"""democracy_exe: A Python package for gooby things."""

from __future__ import annotations

import logging

from democracy_exe.__version__ import __version__


logging.getLogger("asyncio").setLevel(logging.DEBUG)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("faker").setLevel(logging.DEBUG)
logging.getLogger("sentry_sdk").setLevel(logging.WARNING)

# # -------------------------------------------------------------------------------
# # pyright: reportMissingTypeStubs=false
# # pylint: disable=no-member
# # pylint: disable=no-value-for-parameter
# # pyright: reportAttributeAccessIssue=false
# # pyright: reportImportCycles=false

# """democracy_exe: A Python package for gooby things."""

# from __future__ import annotations

# from loguru import logger

# import democracy_exe

# from democracy_exe.__version__ import __version__


# logger.disable("democracy_exe")

# # import logging
# # logging.getLogger("asyncio").setLevel(logging.DEBUG)
# # logging.getLogger("httpx").setLevel(logging.DEBUG)
# # logging.getLogger("faker").setLevel(logging.DEBUG)
# # logging.getLogger("sentry_sdk").setLevel(logging.WARNING)
