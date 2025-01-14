"""
This type stub file was generated by pyright.
"""

import os
import io
from typing import Any, Dict, Optional, Union

"""
This type stub file was generated by pyright.
"""
__all__ = ('File', )
class File:
    r"""A parameter object used for :meth:`abc.Messageable.send`
    for sending file objects.

    .. note::

        File objects are single use and are not meant to be reused in
        multiple :meth:`abc.Messageable.send`\s.

    Attributes
    -----------
    fp: Union[:class:`os.PathLike`, :class:`io.BufferedIOBase`]
        A file-like object opened in binary mode and read mode
        or a filename representing a file in the hard drive to
        open.

        .. note::

            If the file-like object passed is opened via ``open`` then the
            modes 'rb' should be used.

            To pass binary data, consider usage of ``io.BytesIO``.

    spoiler: :class:`bool`
        Whether the attachment is a spoiler. If left unspecified, the :attr:`~File.filename` is used
        to determine if the file is a spoiler.
    description: Optional[:class:`str`]
        The file description to display, currently only supported for images.

        .. versionadded:: 2.0
    """
    __slots__ = ...
    def __init__(self, fp: Union[str, bytes, os.PathLike[Any], io.BufferedIOBase], filename: Optional[str] = ..., *, spoiler: bool = ..., description: Optional[str] = ...) -> None:
        ...
    
    @property
    def filename(self) -> str:
        """:class:`str`: The filename to display when uploading to Discord.
        If this is not given then it defaults to ``fp.name`` or if ``fp`` is
        a string then the ``filename`` will default to the string given.
        """
        ...
    
    @filename.setter
    def filename(self, value: str) -> None:
        ...
    
    def reset(self, *, seek: Union[int, bool] = ...) -> None:
        ...
    
    def close(self) -> None:
        ...
    
    def to_dict(self, index: int) -> Dict[str, Any]:
        ...
    


