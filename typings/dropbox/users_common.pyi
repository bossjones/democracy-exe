"""
This type stub file was generated by pyright.
"""

from stone.backends.python_rsrc import stone_base as bb

"""
This namespace contains common data types used within the users namespace.
"""
class AccountType(bb.Union):
    """
    What type of account this user has.

    This class acts as a tagged union. Only one of the ``is_*`` methods will
    return true. To get the associated value of a tag (if one exists), use the
    corresponding ``get_*`` method.

    :ivar users_common.AccountType.basic: The basic account type.
    :ivar users_common.AccountType.pro: The Dropbox Pro account type.
    :ivar users_common.AccountType.business: The Dropbox Business account type.
    """
    _catch_all = ...
    basic = ...
    pro = ...
    business = ...
    def is_basic(self):
        """
        Check if the union tag is ``basic``.

        :rtype: bool
        """
        ...
    
    def is_pro(self):
        """
        Check if the union tag is ``pro``.

        :rtype: bool
        """
        ...
    
    def is_business(self):
        """
        Check if the union tag is ``business``.

        :rtype: bool
        """
        ...
    


AccountType_validator = ...
AccountId_validator = ...
ROUTES = ...