"""
This type stub file was generated by pyright.
"""

from dropbox.base import DropboxBase
from dropbox.base_team import DropboxTeamBase

__all__ = ['Dropbox', 'DropboxTeam', 'create_session']
__version__ = ...
PATH_ROOT_HEADER = ...
HTTP_STATUS_INVALID_PATH_ROOT = ...
TOKEN_EXPIRATION_BUFFER = ...
SELECT_ADMIN_HEADER = ...
SELECT_USER_HEADER = ...
USER_AUTH = ...
TEAM_AUTH = ...
APP_AUTH = ...
NO_AUTH = ...
class RouteResult:
    """The successful result of a call to a route."""
    def __init__(self, obj_result, http_resp=...) -> None:
        """
        :param str obj_result: The result of a route not including the binary
            payload portion, if one exists. Must be serialized JSON.
        :param requests.models.Response http_resp: A raw HTTP response. It will
            be used to stream the binary-body payload of the response.
        """
        ...
    


class RouteErrorResult:
    """The error result of a call to a route."""
    def __init__(self, request_id, obj_result) -> None:
        """
        :param str request_id: A request_id can be shared with Dropbox Support
            to pinpoint the exact request that returns an error.
        :param str obj_result: The result of a route not including the binary
            payload portion, if one exists.
        """
        ...
    


def create_session(max_connections=..., proxies=..., ca_certs=...): # -> Session:
    """
    Creates a session object that can be used by multiple :class:`Dropbox` and
    :class:`DropboxTeam` instances. This lets you share a connection pool
    amongst them, as well as proxy parameters.

    :param int max_connections: Maximum connection pool size.
    :param dict proxies: See the `requests module
            <http://docs.python-requests.org/en/latest/user/advanced/#proxies>`_
            for more details.
    :rtype: :class:`requests.sessions.Session`. `See the requests module
        <http://docs.python-requests.org/en/latest/user/advanced/#session-objects>`_
        for more details.
    """
    ...

class _DropboxTransport:
    """
    Responsible for implementing the wire protocol for making requests to the
    Dropbox API.
    """
    _API_VERSION = ...
    _ROUTE_STYLE_DOWNLOAD = ...
    _ROUTE_STYLE_UPLOAD = ...
    _ROUTE_STYLE_RPC = ...
    def __init__(self, oauth2_access_token=..., max_retries_on_error=..., max_retries_on_rate_limit=..., user_agent=..., session=..., headers=..., timeout=..., oauth2_refresh_token=..., oauth2_access_token_expiration=..., app_key=..., app_secret=..., scope=..., ca_certs=...) -> None:
        """
        :param str oauth2_access_token: OAuth2 access token for making client
            requests.
        :param int max_retries_on_error: On 5xx errors, the number of times to
            retry.
        :param Optional[int] max_retries_on_rate_limit: On 429 errors, the
            number of times to retry. If `None`, always retries.
        :param str user_agent: The user agent to use when making requests. This
            helps us identify requests coming from your application. We
            recommend you use the format "AppName/Version". If set, we append
            "/OfficialDropboxPythonSDKv2/__version__" to the user_agent,
        :param session: If not provided, a new session (connection pool) is
            created. To share a session across multiple clients, use
            :func:`create_session`.
        :type session: :class:`requests.sessions.Session`
        :param dict headers: Additional headers to add to requests.
        :param Optional[float] timeout: Maximum duration in seconds that
            client will wait for any single packet from the
            server. After the timeout the client will give up on
            connection. If `None`, client will wait forever. Defaults
            to 100 seconds.
        :param str oauth2_refresh_token: OAuth2 refresh token for refreshing access token
        :param datetime oauth2_access_token_expiration: Expiration for oauth2_access_token
        :param str app_key: application key of requesting application; used for token refresh
        :param str app_secret: application secret of requesting application; used for token refresh
            Not required if PKCE was used to authorize the token
        :param list scope: list of scopes to request on refresh.  If left blank,
            refresh will request all available scopes for application
        :param str ca_certs: a path to a file of concatenated CA certificates in PEM format.
            Has the same meaning as when using :func:`ssl.wrap_socket`.
        """
        ...
    
    def clone(self, oauth2_access_token=..., max_retries_on_error=..., max_retries_on_rate_limit=..., user_agent=..., session=..., headers=..., timeout=..., oauth2_refresh_token=..., oauth2_access_token_expiration=..., app_key=..., app_secret=..., scope=...): # -> Self:
        """
        Creates a new copy of the Dropbox client with the same defaults unless modified by
        arguments to clone()

        See constructor for original parameter descriptions.

        :return: New instance of Dropbox client
        :rtype: Dropbox
        """
        ...
    
    def request(self, route, namespace, request_arg, request_binary, timeout=...): # -> tuple[datetime | bytes | Any | list | dict | None, Response | None] | datetime | bytes | list | dict | None:
        """
        Makes a request to the Dropbox API and in the process validates that
        the route argument and result are the expected data types. The
        request_arg is converted to JSON based on the arg_data_type. Likewise,
        the response is deserialized from JSON and converted to an object based
        on the {result,error}_data_type.

        :param host: The Dropbox API host to connect to.
        :param route: The route to make the request to.
        :type route: :class:`stone.backends.python_rsrc.stone_base.Route`
        :param request_arg: Argument for the route that conforms to the
            validator specified by route.arg_type.
        :param request_binary: String or file pointer representing the binary
            payload. Use None if there is no binary payload.
        :param Optional[float] timeout: Maximum duration in seconds
            that client will wait for any single packet from the
            server. After the timeout the client will give up on
            connection. If `None`, will use default timeout set on
            Dropbox object.  Defaults to `None`.
        :return: The route's result.
        """
        ...
    
    def check_and_refresh_access_token(self): # -> None:
        """
        Checks if access token needs to be refreshed and refreshes if possible
        :return:
        """
        ...
    
    def refresh_access_token(self, host=..., scope=...): # -> None:
        """
        Refreshes an access token via refresh token if available

        :param host: host to hit token endpoint with
        :param scope: list of permission scopes for access token
        :return:
        """
        ...
    
    def request_json_object(self, host, route_name, route_style, request_arg, auth_type, request_binary, timeout=...): # -> tuple[Any, Response] | Any:
        """
        Makes a request to the Dropbox API, taking a JSON-serializable Python
        object as an argument, and returning one as a response.

        :param host: The Dropbox API host to connect to.
        :param route_name: The name of the route to invoke.
        :param route_style: The style of the route.
        :param str request_arg: A JSON-serializable Python object representing
            the argument for the route.
        :param auth_type str
        :param Optional[bytes] request_binary: Bytes representing the binary
            payload. Use None if there is no binary payload.
        :param Optional[float] timeout: Maximum duration in seconds
            that client will wait for any single packet from the
            server. After the timeout the client will give up on
            connection. If `None`, will use default timeout set on
            Dropbox object.  Defaults to `None`.
        :return: The route's result as a JSON-serializable Python object.
        """
        ...
    
    def request_json_string_with_retry(self, host, route_name, route_style, request_json_arg, auth_type, request_binary, timeout=...): # -> RouteErrorResult | RouteResult:
        """
        See :meth:`request_json_object` for description of parameters.

        :param request_json_arg: A string representing the serialized JSON
            argument to the route.
        """
        ...
    
    def request_json_string(self, host, func_name, route_style, request_json_arg, auth_type, request_binary, timeout=...): # -> RouteErrorResult | RouteResult:
        """
        See :meth:`request_json_string_with_retry` for description of
        parameters.
        """
        ...
    
    def raise_dropbox_error_for_resp(self, res): # -> None:
        """Checks for errors from a res and handles appropiately.

        :param res: Response of an api request.
        """
        ...
    
    def with_path_root(self, path_root): # -> Self:
        """
        Creates a clone of the Dropbox instance with the Dropbox-API-Path-Root header
        as the appropriate serialized instance of PathRoot.

        For more information, see
        https://www.dropbox.com/developers/reference/namespace-guide#pathrootmodes

        :param PathRoot path_root: instance of PathRoot to serialize into the headers field
        :return: A :class: `Dropbox`
        :rtype: Dropbox
        """
        ...
    
    def close(self): # -> None:
        """
        Cleans up all resources like the request session/network connection.
        """
        ...
    
    def __enter__(self): # -> Self:
        ...
    
    def __exit__(self, *args): # -> None:
        ...
    


class Dropbox(_DropboxTransport, DropboxBase):
    """
    Use this class to make requests to the Dropbox API using a user's access
    token. Methods of this class are meant to act on the corresponding user's
    Dropbox.
    """
    ...


class DropboxTeam(_DropboxTransport, DropboxTeamBase):
    """
    Use this class to make requests to the Dropbox API using a team's access
    token. Methods of this class are meant to act on the team, but there is
    also an :meth:`as_user` method for assuming a team member's identity.
    """
    def as_admin(self, team_member_id): # -> Dropbox:
        """
        Allows a team credential to assume the identity of an administrator on the team
        and perform operations on any team-owned content.

        :param str team_member_id: team member id of administrator to perform actions with
        :return: A :class:`Dropbox` object that can be used to query on behalf
            of this admin of the team.
        :rtype: Dropbox
        """
        ...
    
    def as_user(self, team_member_id): # -> Dropbox:
        """
        Allows a team credential to assume the identity of a member of the
        team.

        :param str team_member_id: team member id of team member to perform actions with
        :return: A :class:`Dropbox` object that can be used to query on behalf
            of this member of the team.
        :rtype: Dropbox
        """
        ...
    


class BadInputException(Exception):
    """
    Thrown if incorrect types/values are used

    This should only ever be thrown during testing, app should have validation of input prior to
    reaching this point
    """
    ...


