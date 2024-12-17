"""
This type stub file was generated by pyright.
"""

class MinimalSet:
    """
    Find contexts where more than one possible target value can
    appear.  E.g. if targets are word-initial letters, and contexts
    are the remainders of words, then we would like to find cases like
    "fat" vs "cat", and "training" vs "draining".  If targets are
    parts-of-speech and contexts are words, then we would like to find
    cases like wind (noun) 'air in rapid motion', vs wind (verb)
    'coil, wrap'.
    """
    def __init__(self, parameters=...) -> None:
        """
        Create a new minimal set.

        :param parameters: The (context, target, display) tuples for the item
        :type parameters: list(tuple(str, str, str))
        """
        ...
    
    def add(self, context, target, display): # -> None:
        """
        Add a new item to the minimal set, having the specified
        context, target, and display form.

        :param context: The context in which the item of interest appears
        :type context: str
        :param target: The item of interest
        :type target: str
        :param display: The information to be reported for each item
        :type display: str
        """
        ...
    
    def contexts(self, minimum=...): # -> list[Any]:
        """
        Determine which contexts occurred with enough distinct targets.

        :param minimum: the minimum number of distinct target forms
        :type minimum: int
        :rtype: list
        """
        ...
    
    def display(self, context, target, default=...): # -> str:
        ...
    
    def display_all(self, context): # -> list[Any]:
        ...
    
    def targets(self): # -> set[Any]:
        ...
    


