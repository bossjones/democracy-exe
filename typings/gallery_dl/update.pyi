"""
This type stub file was generated by pyright.
"""

from .extractor.common import Extractor
from .job import DownloadJob

REPOS = ...
BINARIES_STABLE = ...
BINARIES_DEV = ...
BINARIES = ...
class UpdateJob(DownloadJob):
    def handle_url(self, url, kwdict): # -> None:
        ...
    


class UpdateExtractor(Extractor):
    category = ...
    root = ...
    root_api = ...
    pattern = ...
    def items(self): # -> Generator[tuple[int, Any] | tuple[int, str, Any], Any, None]:
        ...
    

