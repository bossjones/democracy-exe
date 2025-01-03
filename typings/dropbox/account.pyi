"""
This type stub file was generated by pyright.
"""

from stone.backends.python_rsrc import stone_base as bb

class PhotoSourceArg(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar str account.PhotoSourceArg.base64_data: Image data in base64-encoded
        bytes.
    """
    _catch_all = ...
    other = ...
    @classmethod
    def base64_data(cls, val): # -> Self:
        """
        Create an instance of this class set to the ``base64_data`` tag with
        value ``val``.

        :param str val:
        :rtype: PhotoSourceArg
        """
        ...
    
    def is_base64_data(self):
        """
        Check if the union tag is ``base64_data``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    
    def get_base64_data(self): # -> None:
        """
        Image data in base64-encoded bytes.

        Only call this if :meth:`is_base64_data` is true.

        :rtype: str
        """
        ...
    


PhotoSourceArg_validator = ...
class SetProfilePhotoArg(bb.Struct):
    """
    :ivar account.SetProfilePhotoArg.photo: Image to set as the user's new
        profile photo.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, photo=...) -> None:
        ...
    
    photo = ...


SetProfilePhotoArg_validator = ...
class SetProfilePhotoError(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar account.SetProfilePhotoError.file_type_error: File cannot be set as
        profile photo.
    :ivar account.SetProfilePhotoError.file_size_error: File cannot exceed 10
        MB.
    :ivar account.SetProfilePhotoError.dimension_error: Image must be larger
        than 128 x 128.
    :ivar account.SetProfilePhotoError.thumbnail_error: Image could not be
        thumbnailed.
    :ivar account.SetProfilePhotoError.transient_error: Temporary infrastructure
        failure, please retry.
    """
    _catch_all = ...
    file_type_error = ...
    file_size_error = ...
    dimension_error = ...
    thumbnail_error = ...
    transient_error = ...
    other = ...
    def is_file_type_error(self):
        """
        Check if the union tag is ``file_type_error``.

        :rtype: bool
        """
        ...
    
    def is_file_size_error(self):
        """
        Check if the union tag is ``file_size_error``.

        :rtype: bool
        """
        ...
    
    def is_dimension_error(self):
        """
        Check if the union tag is ``dimension_error``.

        :rtype: bool
        """
        ...
    
    def is_thumbnail_error(self):
        """
        Check if the union tag is ``thumbnail_error``.

        :rtype: bool
        """
        ...
    
    def is_transient_error(self):
        """
        Check if the union tag is ``transient_error``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


SetProfilePhotoError_validator = ...
class SetProfilePhotoResult(bb.Struct):
    """
    :ivar account.SetProfilePhotoResult.profile_photo_url: URL for the photo
        representing the user, if one is set.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, profile_photo_url=...) -> None:
        ...
    
    profile_photo_url = ...


SetProfilePhotoResult_validator = ...
set_profile_photo = ...
ROUTES = ...