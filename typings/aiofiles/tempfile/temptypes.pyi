"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete, OpenBinaryMode
from asyncio import AbstractEventLoop
from collections.abc import Generator, Iterable
from tempfile import TemporaryDirectory
from typing import TypeVar
from aiofiles.base import AsyncBase as AsyncBase

_T = TypeVar("_T")
class AsyncSpooledTemporaryFile(AsyncBase[_T]):
    def fileno(self) -> Generator[Incomplete, Incomplete, Incomplete]:
        ...
    
    def rollover(self) -> Generator[Incomplete, Incomplete, Incomplete]:
        ...
    
    async def close(self) -> None:
        ...
    
    async def flush(self) -> None:
        ...
    
    async def isatty(self) -> bool:
        ...
    
    async def read(self, n: int = ..., /):
        ...
    
    async def readline(self, limit: int | None = ..., /):
        ...
    
    async def readlines(self, hint: int = ..., /) -> list[Incomplete]:
        ...
    
    async def seek(self, offset: int, whence: int = ...) -> int:
        ...
    
    async def tell(self) -> int:
        ...
    
    async def truncate(self, size: int | None = ...) -> None:
        ...
    
    @property
    def closed(self) -> bool:
        ...
    
    @property
    def encoding(self) -> str:
        ...
    
    @property
    def mode(self) -> OpenBinaryMode:
        ...
    
    @property
    def name(self) -> str:
        ...
    
    @property
    def newlines(self) -> str:
        ...
    
    async def write(self, s) -> int:
        ...
    
    async def writelines(self, iterable: Iterable[Incomplete]) -> None:
        ...
    


class AsyncTemporaryDirectory:
    async def cleanup(self) -> None:
        ...
    
    @property
    def name(self):
        ...
    
    def __init__(self, file: TemporaryDirectory[Incomplete], loop: AbstractEventLoop | None, executor: Incomplete | None) -> None:
        ...
    
    async def close(self) -> None:
        ...
    


