"""
This type stub file was generated by pyright.
"""

import sys
from _typeshed import BytesPath, Incomplete, OpenBinaryMode, OpenBinaryModeReading, OpenBinaryModeUpdating, OpenBinaryModeWriting, OpenTextMode, StrOrBytesPath, StrPath
from asyncio import AbstractEventLoop
from typing import AnyStr, Literal, overload
from ..base import AiofilesContextManager
from ..threadpool.binary import AsyncBufferedIOBase, AsyncBufferedReader, AsyncFileIO
from ..threadpool.text import AsyncTextIOWrapper
from .temptypes import AsyncTemporaryDirectory

@overload
def TemporaryFile(mode: OpenTextMode, buffering: int = ..., encoding: str | None = ..., newline: str | None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncTextIOWrapper]:
    ...

@overload
def TemporaryFile(mode: OpenBinaryMode, buffering: Literal[0], encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncFileIO]:
    ...

@overload
def TemporaryFile(mode: OpenBinaryModeReading | OpenBinaryModeUpdating = ..., buffering: Literal[-1, 1] = ..., encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncBufferedReader]:
    ...

@overload
def TemporaryFile(mode: OpenBinaryModeWriting, buffering: Literal[-1, 1] = ..., encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncBufferedIOBase]:
    ...

if sys.version_info >= (3, 12):
    @overload
    def NamedTemporaryFile(mode: OpenTextMode, buffering: int = ..., encoding: str | None = ..., newline: str | None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., delete: bool = ..., delete_on_close: bool = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncTextIOWrapper]:
        ...
    
    @overload
    def NamedTemporaryFile(mode: OpenBinaryMode, buffering: Literal[0], encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., delete: bool = ..., delete_on_close: bool = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncFileIO]:
        ...
    
    @overload
    def NamedTemporaryFile(mode: OpenBinaryModeReading | OpenBinaryModeUpdating = ..., buffering: Literal[-1, 1] = ..., encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., delete: bool = ..., delete_on_close: bool = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncBufferedReader]:
        ...
    
    @overload
    def NamedTemporaryFile(mode: OpenBinaryModeWriting, buffering: Literal[-1, 1] = ..., encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., delete: bool = ..., delete_on_close: bool = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncBufferedIOBase]:
        ...
    
else:
    ...
@overload
def SpooledTemporaryFile(max_size: int = ..., *, mode: OpenTextMode, buffering: int = ..., encoding: str | None = ..., newline: str | None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncTextIOWrapper]:
    ...

@overload
def SpooledTemporaryFile(max_size: int, mode: OpenTextMode, buffering: int = ..., encoding: str | None = ..., newline: str | None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncTextIOWrapper]:
    ...

@overload
def SpooledTemporaryFile(max_size: int = ..., mode: OpenBinaryMode = ..., *, buffering: Literal[0], encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncFileIO]:
    ...

@overload
def SpooledTemporaryFile(max_size: int, mode: OpenBinaryMode, buffering: Literal[0], encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncFileIO]:
    ...

@overload
def SpooledTemporaryFile(max_size: int = ..., mode: OpenBinaryModeReading | OpenBinaryModeUpdating = ..., buffering: Literal[-1, 1] = ..., encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncBufferedReader]:
    ...

@overload
def SpooledTemporaryFile(max_size: int = ..., *, mode: OpenBinaryModeWriting, buffering: Literal[-1, 1] = ..., encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncBufferedIOBase]:
    ...

@overload
def SpooledTemporaryFile(max_size: int, mode: OpenBinaryModeWriting, buffering: Literal[-1, 1] = ..., encoding: None = ..., newline: None = ..., suffix: AnyStr | None = ..., prefix: AnyStr | None = ..., dir: StrOrBytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManager[AsyncBufferedIOBase]:
    ...

@overload
def TemporaryDirectory(suffix: str | None = ..., prefix: str | None = ..., dir: StrPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManagerTempDir:
    ...

@overload
def TemporaryDirectory(suffix: bytes | None = ..., prefix: bytes | None = ..., dir: BytesPath | None = ..., loop: AbstractEventLoop | None = ..., executor: Incomplete | None = ...) -> AiofilesContextManagerTempDir:
    ...

class AiofilesContextManagerTempDir(AiofilesContextManager[AsyncTemporaryDirectory]):
    async def __aenter__(self) -> str:
        ...
    


