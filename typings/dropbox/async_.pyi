"""
This type stub file was generated by pyright.
"""

from stone.backends.python_rsrc import stone_base as bb

class LaunchResultBase(bb.Union):
    """
    Result returned by methods that launch an asynchronous job. A method who may
    either launch an asynchronous job, or complete the request synchronously,
    can use this union by extending it, and adding a 'complete' field with the
    type of the synchronous response. See :class:`LaunchEmptyResult` for an
    example.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar str async.LaunchResultBase.async_job_id: This response indicates that
        the processing is asynchronous. The string is an id that can be used to
        obtain the status of the asynchronous job.
    """
    _catch_all = ...
    @classmethod
    def async_job_id(cls, val): # -> Self:
        """
        Create an instance of this class set to the ``async_job_id`` tag with
        value ``val``.

        :param str val:
        :rtype: LaunchResultBase
        """
        ...
    
    def is_async_job_id(self):
        """
        Check if the union tag is ``async_job_id``.

        :rtype: bool
        """
        ...
    
    def get_async_job_id(self): # -> None:
        """
        This response indicates that the processing is asynchronous. The string
        is an id that can be used to obtain the status of the asynchronous job.

        Only call this if :meth:`is_async_job_id` is true.

        :rtype: str
        """
        ...
    


LaunchResultBase_validator = ...
class LaunchEmptyResult(LaunchResultBase):
    """
    Result returned by methods that may either launch an asynchronous job or
    complete synchronously. Upon synchronous completion of the job, no
    additional information is returned.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar async.LaunchEmptyResult.complete: The job finished synchronously and
        successfully.
    """
    complete = ...
    def is_complete(self):
        """
        Check if the union tag is ``complete``.

        :rtype: bool
        """
        ...
    


LaunchEmptyResult_validator = ...
class PollArg(bb.Struct):
    """
    Arguments for methods that poll the status of an asynchronous job.

    :ivar async.PollArg.async_job_id: Id of the asynchronous job. This is the
        value of a response returned from the method that launched the job.
    """
    __slots__ = ...
    _has_required_fields = ...
    def __init__(self, async_job_id=...) -> None:
        ...
    
    async_job_id = ...


PollArg_validator = ...
class PollResultBase(bb.Union):
    """
    Result returned by methods that poll for the status of an asynchronous job.
    Unions that extend this union should add a 'complete' field with a type of
    the information returned upon job completion. See :class:`PollEmptyResult`
    for an example.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar async.PollResultBase.in_progress: The asynchronous job is still in
        progress.
    """
    _catch_all = ...
    in_progress = ...
    def is_in_progress(self):
        """
        Check if the union tag is ``in_progress``.

        :rtype: bool
        """
        ...
    


PollResultBase_validator = ...
class PollEmptyResult(PollResultBase):
    """
    Result returned by methods that poll for the status of an asynchronous job.
    Upon completion of the job, no additional information is returned.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar async.PollEmptyResult.complete: The asynchronous job has completed
        successfully.
    """
    complete = ...
    def is_complete(self):
        """
        Check if the union tag is ``complete``.

        :rtype: bool
        """
        ...
    


PollEmptyResult_validator = ...
class PollError(bb.Union):
    """
    Error returned by methods for polling the status of asynchronous job.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar async.PollError.invalid_async_job_id: The job ID is invalid.
    :ivar async.PollError.internal_error: Something went wrong with the job on
        Dropbox's end. You'll need to verify that the action you were taking
        succeeded, and if not, try again. This should happen very rarely.
    """
    _catch_all = ...
    invalid_async_job_id = ...
    internal_error = ...
    other = ...
    def is_invalid_async_job_id(self):
        """
        Check if the union tag is ``invalid_async_job_id``.

        :rtype: bool
        """
        ...
    
    def is_internal_error(self):
        """
        Check if the union tag is ``internal_error``.

        :rtype: bool
        """
        ...
    
    def is_other(self):
        """
        Check if the union tag is ``other``.

        :rtype: bool
        """
        ...
    


PollError_validator = ...
AsyncJobId_validator = ...
ROUTES = ...