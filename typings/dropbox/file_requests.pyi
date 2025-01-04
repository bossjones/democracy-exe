"""
This type stub file was generated by pyright.
"""

from stone.backends.python_rsrc import stone_base as bb

"""
This namespace contains endpoints and data types for file request operations.
"""
class GeneralFileRequestsError(bb.Union):
    """
    There is an error accessing the file requests functionality.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar file_requests.GeneralFileRequestsError.disabled_for_team: This user's
        Dropbox Business team doesn't allow file requests.
    """
    _catch_all = ...
    disabled_for_team = ...
    other = ...
    def is_disabled_for_team(self):
        """
        Check if the union tag is ``disabled_for_team``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


GeneralFileRequestsError_validator = ...
class CountFileRequestsError(GeneralFileRequestsError):
    """
    There was an error counting the file requests.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.
    """
    ...


CountFileRequestsError_validator = ...
class CountFileRequestsResult(bb.Struct):
    """
    Result for :meth:`dropbox.dropbox_client.Dropbox.file_requests_count`.

    :ivar file_requests.CountFileRequestsResult.file_request_count: The number
        file requests owner by this user.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, file_request_count=...) -> None:
        ...
    
    file_request_count = ...


CountFileRequestsResult_validator = ...
class CreateFileRequestArgs(bb.Struct):
    """
    Arguments for :meth:`dropbox.dropbox_client.Dropbox.file_requests_create`.

    :ivar file_requests.CreateFileRequestArgs.title: The title of the file
        request. Must not be empty.
    :ivar file_requests.CreateFileRequestArgs.destination: The path of the
        folder in the Dropbox where uploaded files will be sent. For apps with
        the app folder permission, this will be relative to the app folder.
    :ivar file_requests.CreateFileRequestArgs.deadline: The deadline for the
        file request. Deadlines can only be set by Professional and Business
        accounts.
    :ivar file_requests.CreateFileRequestArgs.open: Whether or not the file
        request should be open. If the file request is closed, it will not
        accept any file submissions, but it can be opened later.
    :ivar file_requests.CreateFileRequestArgs.description: A description of the
        file request.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, title=..., destination=..., deadline=..., open=..., description=...) -> None:
        ...
    
    title = ...
    destination = ...
    deadline = ...
    open = ...
    description = ...


CreateFileRequestArgs_validator = ...
class FileRequestError(GeneralFileRequestsError):
    """
    There is an error with the file request.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar file_requests.FileRequestError.not_found: This file request ID was not
        found.
    :ivar file_requests.FileRequestError.not_a_folder: The specified path is not
        a folder.
    :ivar file_requests.FileRequestError.app_lacks_access: This file request is
        not accessible to this app. Apps with the app folder permission can only
        access file requests in their app folder.
    :ivar file_requests.FileRequestError.no_permission: This user doesn't have
        permission to access or modify this file request.
    :ivar file_requests.FileRequestError.email_unverified: This user's email
        address is not verified. File requests are only available on accounts
        with a verified email address. Users can verify their email address
        `here <https://www.dropbox.com/help/317>`_.
    :ivar file_requests.FileRequestError.validation_error: There was an error
        validating the request. For example, the title was invalid, or there
        were disallowed characters in the destination path.
    """
    not_found = ...
    not_a_folder = ...
    app_lacks_access = ...
    no_permission = ...
    email_unverified = ...
    validation_error = ...
    def is_not_found(self):
        """
        Check if the union tag is ``not_found``.

        :rtype: bool
        """
        ...
    
    def is_not_a_folder(self):
        """
        Check if the union tag is ``not_a_folder``.

        :rtype: bool
        """
        ...
    
    def is_app_lacks_access(self):
        """
        Check if the union tag is ``app_lacks_access``.

        :rtype: bool
        """
        ...
    
    def is_no_permission(self):
        """
        Check if the union tag is ``no_permission``.

        :rtype: bool
        """
        ...
    
    def is_email_unverified(self):
        """
        Check if the union tag is ``email_unverified``.

        :rtype: bool
        """
        ...
    
    def is_validation_error(self):
        """
        Check if the union tag is ``validation_error``.

        :rtype: bool
        """
        ...
    


FileRequestError_validator = ...
class CreateFileRequestError(FileRequestError):
    """
    There was an error creating the file request.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar file_requests.CreateFileRequestError.invalid_location: File requests
        are not available on the specified folder.
    :ivar file_requests.CreateFileRequestError.rate_limit: The user has reached
        the rate limit for creating file requests. The limit is currently 4000
        file requests total.
    """
    invalid_location = ...
    rate_limit = ...
    def is_invalid_location(self):
        """
        Check if the union tag is ``invalid_location``.

        :rtype: bool
        """
        ...
    
    def is_rate_limit(self):
        """
        Check if the union tag is ``rate_limit``.

        :rtype: bool
        """
        ...
    


CreateFileRequestError_validator = ...
class DeleteAllClosedFileRequestsError(FileRequestError):
    """
    There was an error deleting all closed file requests.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.
    """
    ...


DeleteAllClosedFileRequestsError_validator = ...
class DeleteAllClosedFileRequestsResult(bb.Struct):
    """
    Result for
    :meth:`dropbox.dropbox_client.Dropbox.file_requests_delete_all_closed`.

    :ivar file_requests.DeleteAllClosedFileRequestsResult.file_requests: The
        file requests deleted for this user.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, file_requests=...) -> None:
        ...
    
    file_requests = ...


DeleteAllClosedFileRequestsResult_validator = ...
class DeleteFileRequestArgs(bb.Struct):
    """
    Arguments for :meth:`dropbox.dropbox_client.Dropbox.file_requests_delete`.

    :ivar file_requests.DeleteFileRequestArgs.ids: List IDs of the file requests
        to delete.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, ids=...) -> None:
        ...
    
    ids = ...


DeleteFileRequestArgs_validator = ...
class DeleteFileRequestError(FileRequestError):
    """
    There was an error deleting these file requests.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar file_requests.DeleteFileRequestError.file_request_open: One or more
        file requests currently open.
    """
    file_request_open = ...
    def is_file_request_open(self):
        """
        Check if the union tag is ``file_request_open``.

        :rtype: bool
        """
        ...
    


DeleteFileRequestError_validator = ...
class DeleteFileRequestsResult(bb.Struct):
    """
    Result for :meth:`dropbox.dropbox_client.Dropbox.file_requests_delete`.

    :ivar file_requests.DeleteFileRequestsResult.file_requests: The file
        requests deleted by the request.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, file_requests=...) -> None:
        ...
    
    file_requests = ...


DeleteFileRequestsResult_validator = ...
class FileRequest(bb.Struct):
    """
    A `file request <https://www.dropbox.com/help/9090>`_ for receiving files
    into the user's Dropbox account.

    :ivar file_requests.FileRequest.id: The ID of the file request.
    :ivar file_requests.FileRequest.url: The URL of the file request.
    :ivar file_requests.FileRequest.title: The title of the file request.
    :ivar file_requests.FileRequest.destination: The path of the folder in the
        Dropbox where uploaded files will be sent. This can be None if the
        destination was removed. For apps with the app folder permission, this
        will be relative to the app folder.
    :ivar file_requests.FileRequest.created: When this file request was created.
    :ivar file_requests.FileRequest.deadline: The deadline for this file
        request. Only set if the request has a deadline.
    :ivar file_requests.FileRequest.is_open: Whether or not the file request is
        open. If the file request is closed, it will not accept any more file
        submissions.
    :ivar file_requests.FileRequest.file_count: The number of files this file
        request has received.
    :ivar file_requests.FileRequest.description: A description of the file
        request.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, id=..., url=..., title=..., created=..., is_open=..., file_count=..., destination=..., deadline=..., description=...) -> None:
        ...
    
    id = ...
    url = ...
    title = ...
    destination = ...
    created = ...
    deadline = ...
    is_open = ...
    file_count = ...
    description = ...


FileRequest_validator = ...
class FileRequestDeadline(bb.Struct):
    """
    :ivar file_requests.FileRequestDeadline.deadline: The deadline for this file
        request.
    :ivar file_requests.FileRequestDeadline.allow_late_uploads: If set, allow
        uploads after the deadline has passed. These     uploads will be marked
        overdue.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, deadline=..., allow_late_uploads=...) -> None:
        ...
    
    deadline = ...
    allow_late_uploads = ...


FileRequestDeadline_validator = ...
class GetFileRequestArgs(bb.Struct):
    """
    Arguments for :meth:`dropbox.dropbox_client.Dropbox.file_requests_get`.

    :ivar file_requests.GetFileRequestArgs.id: The ID of the file request to
        retrieve.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, id=...) -> None:
        ...
    
    id = ...


GetFileRequestArgs_validator = ...
class GetFileRequestError(FileRequestError):
    """
    There was an error retrieving the specified file request.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.
    """
    ...


GetFileRequestError_validator = ...
class GracePeriod(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.
    """
    _catch_all = ...
    one_day = ...
    two_days = ...
    seven_days = ...
    thirty_days = ...
    always = ...
    other = ...
    def is_one_day(self):
        """
        Check if the union tag is ``one_day``.

        :rtype: bool
        """
        ...
    
    def is_two_days(self):
        """
        Check if the union tag is ``two_days``.

        :rtype: bool
        """
        ...
    
    def is_seven_days(self):
        """
        Check if the union tag is ``seven_days``.

        :rtype: bool
        """
        ...
    
    def is_thirty_days(self):
        """
        Check if the union tag is ``thirty_days``.

        :rtype: bool
        """
        ...
    
    def is_always(self):
        """
        Check if the union tag is ``always``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


GracePeriod_validator = ...
class ListFileRequestsArg(bb.Struct):
    """
    Arguments for :meth:`dropbox.dropbox_client.Dropbox.file_requests_list`.

    :ivar file_requests.ListFileRequestsArg.limit: The maximum number of file
        requests that should be returned per request.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, limit=...) -> None:
        ...
    
    limit = ...


ListFileRequestsArg_validator = ...
class ListFileRequestsContinueArg(bb.Struct):
    """
    :ivar file_requests.ListFileRequestsContinueArg.cursor: The cursor returned
        by the previous API call specified in the endpoint description.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, cursor=...) -> None:
        ...
    
    cursor = ...


ListFileRequestsContinueArg_validator = ...
class ListFileRequestsContinueError(GeneralFileRequestsError):
    """
    There was an error retrieving the file requests.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar file_requests.ListFileRequestsContinueError.invalid_cursor: The cursor
        is invalid.
    """
    invalid_cursor = ...
    def is_invalid_cursor(self):
        """
        Check if the union tag is ``invalid_cursor``.

        :rtype: bool
        """
        ...
    


ListFileRequestsContinueError_validator = ...
class ListFileRequestsError(GeneralFileRequestsError):
    """
    There was an error retrieving the file requests.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.
    """
    ...


ListFileRequestsError_validator = ...
class ListFileRequestsResult(bb.Struct):
    """
    Result for :meth:`dropbox.dropbox_client.Dropbox.file_requests_list`.

    :ivar file_requests.ListFileRequestsResult.file_requests: The file requests
        owned by this user. Apps with the app folder permission will only see
        file requests in their app folder.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, file_requests=...) -> None:
        ...
    
    file_requests = ...


ListFileRequestsResult_validator = ...
class ListFileRequestsV2Result(bb.Struct):
    """
    Result for :meth:`dropbox.dropbox_client.Dropbox.file_requests_list` and
    :meth:`dropbox.dropbox_client.Dropbox.file_requests_list_continue`.

    :ivar file_requests.ListFileRequestsV2Result.file_requests: The file
        requests owned by this user. Apps with the app folder permission will
        only see file requests in their app folder.
    :ivar file_requests.ListFileRequestsV2Result.cursor: Pass the cursor into
        :meth:`dropbox.dropbox_client.Dropbox.file_requests_list_continue` to
        obtain additional file requests.
    :ivar file_requests.ListFileRequestsV2Result.has_more: Is true if there are
        additional file requests that have not been returned yet. An additional
        call to :route:list/continue` can retrieve them.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, file_requests=..., cursor=..., has_more=...) -> None:
        ...
    
    file_requests = ...
    cursor = ...
    has_more = ...


ListFileRequestsV2Result_validator = ...
class UpdateFileRequestArgs(bb.Struct):
    """
    Arguments for :meth:`dropbox.dropbox_client.Dropbox.file_requests_update`.

    :ivar file_requests.UpdateFileRequestArgs.id: The ID of the file request to
        update.
    :ivar file_requests.UpdateFileRequestArgs.title: The new title of the file
        request. Must not be empty.
    :ivar file_requests.UpdateFileRequestArgs.destination: The new path of the
        folder in the Dropbox where uploaded files will be sent. For apps with
        the app folder permission, this will be relative to the app folder.
    :ivar file_requests.UpdateFileRequestArgs.deadline: The new deadline for the
        file request. Deadlines can only be set by Professional and Business
        accounts.
    :ivar file_requests.UpdateFileRequestArgs.open: Whether to set this file
        request as open or closed.
    :ivar file_requests.UpdateFileRequestArgs.description: The description of
        the file request.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, id=..., title=..., destination=..., deadline=..., open=..., description=...) -> None:
        ...
    
    id = ...
    title = ...
    destination = ...
    deadline = ...
    open = ...
    description = ...


UpdateFileRequestArgs_validator = ...
class UpdateFileRequestDeadline(bb.Union):
    """
    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar file_requests.UpdateFileRequestDeadline.no_update: Do not change the
        file request's deadline.
    :ivar Optional[FileRequestDeadline]
        file_requests.UpdateFileRequestDeadline.update: If :val:`null`, the file
        request's deadline is cleared.
    """
    _catch_all = ...
    no_update = ...
    other = ...
    @classmethod
    def update(cls, val): # -> Self:
        """
        Create an instance of this class set to the ``update`` tag with value
        ``val``.

        :param FileRequestDeadline val:
        :rtype: UpdateFileRequestDeadline
        """
        ...
    
    def is_no_update(self):
        """
        Check if the union tag is ``no_update``.

        :rtype: bool
        """
        ...
    
    def is_update(self):
        """
        Check if the union tag is ``update``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    
    def get_update(self): # -> None:
        """
        If None, the file request's deadline is cleared.

        Only call this if :meth:`is_update` is true.

        :rtype: FileRequestDeadline
        """
        ...
    


UpdateFileRequestDeadline_validator = ...
class UpdateFileRequestError(FileRequestError):
    """
    There is an error updating the file request.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.
    """
    ...


UpdateFileRequestError_validator = ...
FileRequestId_validator = ...
FileRequestValidationError_validator = ...
count = ...
create = ...
delete = ...
delete_all_closed = ...
get = ...
list_v2 = ...
list = ...
list_continue = ...
update = ...
ROUTES = ...
