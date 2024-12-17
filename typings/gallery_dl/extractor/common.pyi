"""
This type stub file was generated by pyright.
"""

from requests.adapters import HTTPAdapter

"""Common classes and constants used by extractor modules."""
urllib3 = ...
class Extractor:
    category = ...
    subcategory = ...
    basecategory = ...
    categorytransfer = ...
    directory_fmt = ...
    filename_fmt = ...
    archive_fmt = ...
    root = ...
    cookies_domain = ...
    cookies_index = ...
    referer = ...
    ciphers = ...
    tls12 = ...
    browser = ...
    request_interval = ...
    request_interval_min = ...
    request_interval_429 = ...
    request_timestamp = ...
    def __init__(self, match) -> None:
        ...
    
    @classmethod
    def from_url(cls, url): # -> Self | None:
        ...
    
    def __iter__(self): # -> Generator[tuple[int, Literal[1]], Any, None]:
        ...
    
    def initialize(self): # -> None:
        ...
    
    def finalize(self): # -> None:
        ...
    
    def items(self): # -> Generator[tuple[int, Literal[1]], Any, None]:
        ...
    
    def skip(self, num): # -> Literal[0]:
        ...
    
    def config(self, key, default=...): # -> None:
        ...
    
    def config2(self, key, key2, default=..., sentinel=...): # -> None:
        ...
    
    def config_deprecated(self, key, deprecated, default=..., sentinel=..., history=...): # -> None:
        ...
    
    def config_accumulate(self, key): # -> list[Any]:
        ...
    
    def config_instance(self, key, default=...): # -> None:
        ...
    
    def request(self, url, method=..., session=..., retries=..., retry_codes=..., encoding=..., fatal=..., notfound=..., **kwargs):
        ...
    
    _handle_429 = ...
    def wait(self, seconds=..., until=..., adjust=..., reason=...): # -> None:
        ...
    
    def sleep(self, seconds, reason): # -> None:
        ...
    
    def input(self, prompt, echo=...): # -> str | None:
        ...
    
    def cookies_load(self, cookies_source): # -> None:
        ...
    
    def cookies_store(self): # -> None:
        """Store the session's cookies in a cookies.txt file"""
        ...
    
    def cookies_update(self, cookies, domain=...): # -> None:
        """Update the session's cookiejar with 'cookies'"""
        ...
    
    def cookies_update_dict(self, cookiedict, domain): # -> None:
        """Update cookiejar with name-value pairs from a dict"""
        ...
    
    def cookies_check(self, cookies_names, domain=...): # -> bool:
        """Check if all 'cookies_names' are in the session's cookiejar"""
        ...
    


class GalleryExtractor(Extractor):
    subcategory = ...
    filename_fmt = ...
    directory_fmt = ...
    archive_fmt = ...
    enum = ...
    def __init__(self, match, url=...) -> None:
        ...
    
    def items(self): # -> Generator[tuple[int, None] | tuple[int, Any, None], Any, None]:
        ...
    
    def login(self): # -> None:
        """Login and set necessary cookies"""
        ...
    
    def metadata(self, page): # -> None:
        """Return a dict with general metadata"""
        ...
    
    def images(self, page): # -> None:
        """Return a list of all (image-url, metadata)-tuples"""
        ...
    


class ChapterExtractor(GalleryExtractor):
    subcategory = ...
    directory_fmt = ...
    filename_fmt = ...
    archive_fmt = ...
    enum = ...


class MangaExtractor(Extractor):
    subcategory = ...
    categorytransfer = ...
    chapterclass = ...
    reverse = ...
    def __init__(self, match, url=...) -> None:
        ...
    
    def items(self): # -> Generator[tuple[int, Any, Any], Any, None]:
        ...
    
    def login(self): # -> None:
        """Login and set necessary cookies"""
        ...
    
    def chapters(self, page): # -> None:
        """Return a list of all (chapter-url, metadata)-tuples"""
        ...
    


class AsynchronousMixin:
    """Run info extraction in a separate thread"""
    def __iter__(self): # -> Generator[Any, Any, None]:
        ...
    
    def async_items(self, messages): # -> None:
        ...
    


class BaseExtractor(Extractor):
    instances = ...
    def __init__(self, match) -> None:
        ...
    
    @classmethod
    def update(cls, instances): # -> str:
        ...
    


class RequestsAdapter(HTTPAdapter):
    def __init__(self, ssl_context=..., source_address=...) -> None:
        ...
    
    def init_poolmanager(self, *args, **kwargs):
        ...
    
    def proxy_manager_for(self, *args, **kwargs):
        ...
    


_adapter_cache = ...
_browser_cookies = ...
HTTP_HEADERS = ...
SSL_CIPHERS = ...
BROTLI = ...
ZSTD = ...
action = ...
if action:
    ...