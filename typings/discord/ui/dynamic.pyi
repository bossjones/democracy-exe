"""
This type stub file was generated by pyright.
"""

import re
from typing import Any, ClassVar, Dict, Generic, Optional, TYPE_CHECKING, Tuple, Type, Union
from .item import Item
from .._types import ClientT
from typing_extensions import Self
from ..interactions import Interaction
from ..components import Component
from ..enums import ComponentType

"""
This type stub file was generated by pyright.
"""
__all__ = ('DynamicItem', )
BaseT = ...
if TYPE_CHECKING:
    V = ...
else:
    ...
class DynamicItem(Generic[BaseT], Item['View']):
    """Represents an item with a dynamic ``custom_id`` that can be used to store state within
    that ``custom_id``.

    The ``custom_id`` parsing is done using the ``re`` module by passing a ``template``
    parameter to the class parameter list.

    This item is generated every time the component is dispatched. This means that
    any variable that holds an instance of this class will eventually be out of date
    and should not be used long term. Their only purpose is to act as a "template"
    for the actual dispatched item.

    When this item is generated, :attr:`view` is set to a regular :class:`View` instance
    from the original message given from the interaction. This means that custom view
    subclasses cannot be accessed from this item.

    .. versionadded:: 2.4

    Parameters
    ------------
    item: :class:`Item`
        The item to wrap with dynamic custom ID parsing.
    template: Union[:class:`str`, ``re.Pattern``]
        The template to use for parsing the ``custom_id``. This can be a string or a compiled
        regular expression. This must be passed as a keyword argument to the class creation.
    row: Optional[:class:`int`]
        The relative row this button belongs to. A Discord component can only have 5
        rows. By default, items are arranged automatically into those 5 rows. If you'd
        like to control the relative positioning of the row then passing an index is advised.
        For example, row=1 will show up before row=2. Defaults to ``None``, which is automatic
        ordering. The row number must be between 0 and 4 (i.e. zero indexed).

    Attributes
    -----------
    item: :class:`Item`
        The item that is wrapped with dynamic custom ID parsing.
    """
    __item_repr_attributes__: Tuple[str, ...] = ...
    __discord_ui_compiled_template__: ClassVar[re.Pattern[str]]
    def __init_subclass__(cls, *, template: Union[str, re.Pattern[str]]) -> None:
        ...
    
    def __init__(self, item: BaseT, *, row: Optional[int] = ...) -> None:
        ...
    
    @property
    def template(self) -> re.Pattern[str]:
        """``re.Pattern``: The compiled regular expression that is used to parse the ``custom_id``."""
        ...
    
    def to_component_dict(self) -> Dict[str, Any]:
        ...
    
    @classmethod
    def from_component(cls: Type[Self], component: Component) -> Self:
        ...
    
    @property
    def type(self) -> ComponentType:
        ...
    
    def is_dispatchable(self) -> bool:
        ...
    
    def is_persistent(self) -> bool:
        ...
    
    @property
    def custom_id(self) -> str:
        """:class:`str`: The ID of the dynamic item that gets received during an interaction."""
        ...
    
    @custom_id.setter
    def custom_id(self, value: str) -> None:
        ...
    
    @property
    def row(self) -> Optional[int]:
        ...
    
    @row.setter
    def row(self, value: Optional[int]) -> None:
        ...
    
    @property
    def width(self) -> int:
        ...
    
    @classmethod
    async def from_custom_id(cls: Type[Self], interaction: Interaction[ClientT], item: Item[Any], match: re.Match[str], /) -> Self:
        """|coro|

        A classmethod that is called when the ``custom_id`` of a component matches the
        ``template`` of the class. This is called when the component is dispatched.

        It must return a new instance of the :class:`DynamicItem`.

        Subclasses *must* implement this method.

        Exceptions raised in this method are logged and ignored.

        .. warning::

            This method is called before the callback is dispatched, therefore
            it means that it is subject to the same timing restrictions as the callback.
            Ergo, you must reply to an interaction within 3 seconds of it being
            dispatched.

        Parameters
        ------------
        interaction: :class:`~discord.Interaction`
            The interaction that the component belongs to.
        item: :class:`~discord.ui.Item`
            The base item that is being dispatched.
        match: ``re.Match``
            The match object that was created from the ``template``
            matching the ``custom_id``.

        Returns
        --------
        :class:`DynamicItem`
            The new instance of the :class:`DynamicItem` with information
            from the ``match`` object.
        """
        ...
    
    async def callback(self, interaction: Interaction[ClientT]) -> Any:
        ...
    
    async def interaction_check(self, interaction: Interaction[ClientT], /) -> bool:
        ...
    


