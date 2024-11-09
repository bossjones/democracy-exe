"""
This type stub file was generated by pyright.
"""

from nltk.sem.logic import (
    AbstractVariableExpression,
    ApplicationExpression,
    BinaryExpression,
    BooleanExpression,
    ConstantExpression,
    EqualityExpression,
    EventVariableExpression,
    Expression,
    FunctionVariableExpression,
    IndividualVariableExpression,
    LambdaExpression,
    LogicParser,
    NegatedExpression,
    OrExpression,
    Tokens,
)

class DrtTokens(Tokens):
    DRS = ...
    DRS_CONC = ...
    PRONOUN = ...
    OPEN_BRACKET = ...
    CLOSE_BRACKET = ...
    COLON = ...
    PUNCT = ...
    SYMBOLS = ...
    TOKENS = ...


class DrtParser(LogicParser):
    """A lambda calculus expression parser."""
    def __init__(self) -> None:
        ...

    def get_all_symbols(self): # -> list[str]:
        """This method exists to be overridden"""
        ...

    def isvariable(self, tok): # -> bool:
        ...

    def handle(self, tok, context): # -> NegatedExpression | LambdaExpression | DRS | DrtProposition | ApplicationExpression | IndividualVariableExpression | FunctionVariableExpression | EventVariableExpression | ConstantExpression | None:
        """This method is intended to be overridden for logics that
        use different operators or expressions"""
        ...

    def make_NegatedExpression(self, expression): # -> DrtNegatedExpression:
        ...

    def handle_DRS(self, tok, context): # -> DRS:
        ...

    def handle_refs(self): # -> list:
        ...

    def handle_conds(self, context): # -> list:
        ...

    def handle_prop(self, tok, context): # -> DrtProposition:
        ...

    def make_EqualityExpression(self, first, second): # -> DrtEqualityExpression:
        """This method serves as a hook for other logic parsers that
        have different equality expression classes"""
        ...

    def get_BooleanExpression_factory(self, tok): # -> Callable[..., DrtConcatenation] | type[DrtOrExpression] | Callable[..., DRS | DrtConcatenation] | None:
        """This method serves as a hook for other logic parsers that
        have different boolean operators"""
        ...

    def make_BooleanExpression(self, factory, first, second):
        ...

    def make_ApplicationExpression(self, function, argument): # -> DrtApplicationExpression:
        ...

    def make_VariableExpression(self, name): # -> DrtIndividualVariableExpression | DrtFunctionVariableExpression | DrtEventVariableExpression | DrtConstantExpression:
        ...

    def make_LambdaExpression(self, variables, term): # -> DrtLambdaExpression:
        ...



class DrtExpression:
    """
    This is the base abstract DRT Expression from which every DRT
    Expression extends.
    """
    _drt_parser = ...
    @classmethod
    def fromstring(cls, s): # -> AndExpression | IffExpression | ImpExpression | OrExpression:
        ...

    def applyto(self, other): # -> DrtApplicationExpression:
        ...

    def __neg__(self): # -> DrtNegatedExpression:
        ...

    def __and__(self, other): # -> _NotImplementedType:
        ...

    def __or__(self, other): # -> DrtOrExpression:
        ...

    def __gt__(self, other) -> bool:
        ...

    def equiv(self, other, prover=...):
        """
        Check for logical equivalence.
        Pass the expression (self <-> other) to the theorem prover.
        If the prover says it is valid, then the self and other are equal.

        :param other: an ``DrtExpression`` to check equality against
        :param prover: a ``nltk.inference.api.Prover``
        """
        ...

    @property
    def type(self):
        ...

    def typecheck(self, signature=...):
        ...

    def __add__(self, other): # -> DrtConcatenation:
        ...

    def get_refs(self, recursive=...):
        """
        Return the set of discourse referents in this DRS.
        :param recursive: bool Also find discourse referents in subterms?
        :return: list of ``Variable`` objects
        """
        ...

    def is_pronoun_function(self): # -> bool:
        """Is self of the form "PRO(x)"?"""
        ...

    def make_EqualityExpression(self, first, second): # -> DrtEqualityExpression:
        ...

    def make_VariableExpression(self, variable): # -> DrtIndividualVariableExpression | DrtFunctionVariableExpression | DrtEventVariableExpression | DrtConstantExpression:
        ...

    def resolve_anaphora(self): # -> ApplicationExpression | DRS | AbstractVariableExpression | NegatedExpression | DrtConcatenation | BinaryExpression | LambdaExpression | None:
        ...

    def eliminate_equality(self):
        ...

    def pretty_format(self): # -> str:
        """
        Draw the DRS
        :return: the pretty print string
        """
        ...

    def pretty_print(self): # -> None:
        ...

    def draw(self): # -> None:
        ...



class DRS(DrtExpression, Expression):
    """A Discourse Representation Structure."""
    def __init__(self, refs, conds, consequent=...) -> None:
        """
        :param refs: list of ``DrtIndividualVariableExpression`` for the
            discourse referents
        :param conds: list of ``Expression`` for the conditions
        """
        ...

    def replace(self, variable, expression, replace_bound=..., alpha_convert=...): # -> Self | DRS:
        """Replace all instances of variable v with expression E in self,
        where v is free in self."""
        ...

    def free(self): # -> Any:
        """:see: Expression.free()"""
        ...

    def get_refs(self, recursive=...): # -> Any:
        """:see: AbstractExpression.get_refs()"""
        ...

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        ...

    def visit_structured(self, function, combinator):
        """:see: Expression.visit_structured()"""
        ...

    def eliminate_equality(self): # -> DRS:
        ...

    def fol(self): # -> ImpExpression | AllExpression | AndExpression | ExistsExpression:
        ...

    def __eq__(self, other) -> bool:
        r"""Defines equality modulo alphabetic variance.
        If we are comparing \x.M  and \y.N, then check equality of M and N[x/y]."""
        ...

    def __ne__(self, other) -> bool:
        ...

    __hash__ = ...
    def __str__(self) -> str:
        ...



def DrtVariableExpression(variable): # -> DrtIndividualVariableExpression | DrtFunctionVariableExpression | DrtEventVariableExpression | DrtConstantExpression:
    """
    This is a factory method that instantiates and returns a subtype of
    ``DrtAbstractVariableExpression`` appropriate for the given variable.
    """
    ...

class DrtAbstractVariableExpression(DrtExpression, AbstractVariableExpression):
    def fol(self): # -> Self:
        ...

    def get_refs(self, recursive=...): # -> list:
        """:see: AbstractExpression.get_refs()"""
        ...

    def eliminate_equality(self): # -> Self:
        ...



class DrtIndividualVariableExpression(DrtAbstractVariableExpression, IndividualVariableExpression):
    ...


class DrtFunctionVariableExpression(DrtAbstractVariableExpression, FunctionVariableExpression):
    ...


class DrtEventVariableExpression(DrtIndividualVariableExpression, EventVariableExpression):
    ...


class DrtConstantExpression(DrtAbstractVariableExpression, ConstantExpression):
    ...


class DrtProposition(DrtExpression, Expression):
    def __init__(self, variable, drs) -> None:
        ...

    def replace(self, variable, expression, replace_bound=..., alpha_convert=...): # -> DrtProposition:
        ...

    def eliminate_equality(self): # -> DrtProposition:
        ...

    def get_refs(self, recursive=...): # -> list:
        ...

    def __eq__(self, other) -> bool:
        ...

    def __ne__(self, other) -> bool:
        ...

    __hash__ = ...
    def fol(self):
        ...

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        ...

    def visit_structured(self, function, combinator):
        """:see: Expression.visit_structured()"""
        ...

    def __str__(self) -> str:
        ...



class DrtNegatedExpression(DrtExpression, NegatedExpression):
    def fol(self): # -> NegatedExpression:
        ...

    def get_refs(self, recursive=...):
        """:see: AbstractExpression.get_refs()"""
        ...



class DrtLambdaExpression(DrtExpression, LambdaExpression):
    def alpha_convert(self, newvar): # -> Self:
        """Rename all occurrences of the variable introduced by this variable
        binder in the expression to ``newvar``.
        :param newvar: ``Variable``, for the new variable
        """
        ...

    def fol(self): # -> LambdaExpression:
        ...

    def get_refs(self, recursive=...): # -> list[Variable]:
        """:see: AbstractExpression.get_refs()"""
        ...



class DrtBinaryExpression(DrtExpression, BinaryExpression):
    def get_refs(self, recursive=...): # -> list:
        """:see: AbstractExpression.get_refs()"""
        ...



class DrtBooleanExpression(DrtBinaryExpression, BooleanExpression):
    ...


class DrtOrExpression(DrtBooleanExpression, OrExpression):
    def fol(self): # -> OrExpression:
        ...



class DrtEqualityExpression(DrtBinaryExpression, EqualityExpression):
    def fol(self): # -> EqualityExpression:
        ...



class DrtConcatenation(DrtBooleanExpression):
    """DRS of the form '(DRS + DRS)'"""
    def __init__(self, first, second, consequent=...) -> None:
        ...

    def replace(self, variable, expression, replace_bound=..., alpha_convert=...): # -> Self:
        """Replace all instances of variable v with expression E in self,
        where v is free in self."""
        ...

    def eliminate_equality(self): # -> DRS:
        ...

    def simplify(self): # -> DRS | Self:
        ...

    def get_refs(self, recursive=...):
        """:see: AbstractExpression.get_refs()"""
        ...

    def getOp(self): # -> Literal['+']:
        ...

    def __eq__(self, other) -> bool:
        r"""Defines equality modulo alphabetic variance.
        If we are comparing \x.M  and \y.N, then check equality of M and N[x/y]."""
        ...

    def __ne__(self, other) -> bool:
        ...

    __hash__ = ...
    def fol(self): # -> ImpExpression | AndExpression:
        ...

    def visit(self, function, combinator):
        """:see: Expression.visit()"""
        ...

    def __str__(self) -> str:
        ...



class DrtApplicationExpression(DrtExpression, ApplicationExpression):
    def fol(self): # -> ApplicationExpression:
        ...

    def get_refs(self, recursive=...): # -> list:
        """:see: AbstractExpression.get_refs()"""
        ...



class PossibleAntecedents(list, DrtExpression, Expression):
    def free(self): # -> set:
        """Set of free variables."""
        ...

    def replace(self, variable, expression, replace_bound=..., alpha_convert=...): # -> PossibleAntecedents:
        """Replace all instances of variable v with expression E in self,
        where v is free in self."""
        ...

    def __str__(self) -> str:
        ...



class AnaphoraResolutionException(Exception):
    ...


def resolve_anaphora(expression, trail=...): # -> ApplicationExpression | DRS | AbstractVariableExpression | NegatedExpression | DrtConcatenation | BinaryExpression | LambdaExpression | None:
    ...

class DrsDrawer:
    BUFFER = ...
    TOPSPACE = ...
    OUTERSPACE = ...
    def __init__(self, drs, size_canvas=..., canvas=...) -> None:
        """
        :param drs: ``DrtExpression``, The DRS to be drawn
        :param size_canvas: bool, True if the canvas size should be the exact size of the DRS
        :param canvas: ``Canvas`` The canvas on which to draw the DRS.  If none is given, create a new canvas.
        """
        ...

    def draw(self, x=..., y=...): # -> tuple | None:
        """Draw the DRS"""
        ...



def demo(): # -> None:
    ...

def test_draw(): # -> None:
    ...

if __name__ == "__main__":
    ...
