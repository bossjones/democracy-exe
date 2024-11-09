"""
This type stub file was generated by pyright.
"""

"""
Finding (and Replacing) Nemo

Instant Regular Expressions
Created by Aristide Grange
"""
windowTitle = ...
initialFind = ...
initialRepl = ...
initialText = ...
images = ...
colors = ...
emphColors = ...
fieldParams = ...
textParams = ...
class Zone:
    def __init__(self, image, initialField, initialText) -> None:
        ...

    def initScrollText(self, frm, txt, contents): # -> None:
        ...

    def refresh(self): # -> None:
        ...



class FindZone(Zone):
    def addTags(self, m): # -> None:
        ...

    def substitute(self, *args): # -> None:
        ...



class ReplaceZone(Zone):
    def addTags(self, m): # -> None:
        ...

    def substitute(self): # -> None:
        ...



def launchRefresh(_): # -> None:
    ...

def app(): # -> None:
    ...

if __name__ == "__main__":
    ...
__all__ = ["app"]
