"""
This type stub file was generated by pyright.
"""

def rm_nl(s):
    ...

class TabEncoder:
    def list2txt(self, s): # -> str:
        ...

    def set2txt(self, s): # -> str:
        ...

    def tup2tab(self, tup): # -> str:
        ...

    def tups2tab(self, x): # -> str:
        ...

    def dict2tab(self, d): # -> str:
        ...

    def ivdict2tab(self, d): # -> str:
        ...



class TabDecoder:
    def txt2list(self, f): # -> list:
        ...

    def txt2set(self, f): # -> set:
        ...

    def tab2tup(self, s): # -> tuple:
        ...

    def tab2tups(self, f): # -> list[tuple]:
        ...

    def tab2dict(self, f): # -> dict:
        ...

    def tab2ivdict(self, f): # -> dict[Any, int]:
        ...



class MaxentEncoder(TabEncoder):
    def tupdict2tab(self, d): # -> str:
        ...



class MaxentDecoder(TabDecoder):
    def tupkey2dict(self, f): # -> dict[tuple[Any, int | Any | bool | None, Any], int]:
        ...



class PunktDecoder(TabDecoder):
    def tab2intdict(self, f): # -> defaultdict[Any, int]:
        ...
