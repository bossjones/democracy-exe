"""
This type stub file was generated by pyright.
"""

class ErrorInterceptor:
    def __init__(self, should_catch, handler_id) -> None:
        ...
    
    def should_catch(self): # -> Any:
        ...
    
    def print(self, record=..., *, exception=...): # -> None:
        ...
    

