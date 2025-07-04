#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Optional, Union


class PrimitiveSet:
    """
    Class that encapsulates a set of primitives and terminals. Used to generate the individuals.
    """

    def __init__(self, name: str, arity: int = 0, prefix: str = "ARG"):
        self.terminals: list['Union[Primitive, Terminal, type[Ephemeral]]'] = []
        self.primitives: list['Union[Primitive, Terminal, type[Ephemeral]]'] = []
        self.arguments: list[str] = []
        self.context: dict[str, 'Optional[Callable[[], object]]'] = {"__builtins__": None}
        self.mapping: dict[str, 'Union[Primitive, Terminal, type[Ephemeral]]'] = {}
        self.terms_count = 0
        self.prims_count = 0
        self.ephemerals: dict[str, type] = {}

        self.name = name
        self.ret = object
        self.ins = [object] * arity

        for i in range(arity):
            arg_str = f"{prefix}{i}"
            self.arguments.append(arg_str)
            term = Terminal(arg_str, object, symbolic=True)
            self._add(term)
            self.terms_count += 1

    def _add(self, prim: 'Union[Primitive, Terminal, type[Ephemeral]]') -> None:
        # Add the primitive or terminal to the lists
        self.primitives.append(prim)
        self.terminals.append(prim)

        # Map the primitive/terminal to its name
        self.mapping[prim.name] = prim

    def add_primitive(
        self,
        primitive: 'Callable[[], object]',
        arity: int,
        name: 'Optional[str]' = None,
    ) -> None:
        """
        Add a primitive to the set.
        """
        if arity <= 0:
            raise ValueError("Arity should be >= 1.")

        args = [object] * arity

        if name is None:
            name = primitive.__name__

        prim = Primitive(name, args, object)

        if name in self.context and self.context[name] is not primitive:
            raise ValueError("Primitives are required to have a unique name. "
                             "Consider using the argument 'name' to rename "
                             f"your second '{name}' primitive.")

        self._add(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def add_ephemeral_constant(self, name: str, ephemeral: 'Callable[..., object]') -> None:
        """
        Add an ephemeral constant to the set.
        """
        if name not in self.ephemerals:
            class_ = type(
                name,
                (Ephemeral,),
                {'func': staticmethod(ephemeral), 'ret': object, 'name': name},
            )
            self.ephemerals[name] = class_
        else:
            warnings.warn(f"Ephemeral constant '{name}' already exists. Skipping.", stacklevel=2)
            return

        self._add(class_)
        self.terms_count += 1

class Terminal:
    """
    Class that encapsulates terminal primitive in expression.

    Terminals can be values or 0-arity functions.
    """

    __slots__ = ('name', 'value', 'ret', 'symbolic')

    def __init__(self, terminal: object, ret: object, *, symbolic: bool):
        self.value = terminal
        self.ret = ret
        self.name = str(terminal)
        self.symbolic = symbolic

    @property
    def arity(self) -> int:
        return 0

    def format(self) -> str:
        return str(self.value) if self.symbolic else repr(self.value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Terminal):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)

        return NotImplemented

class Primitive:
    """
    Class that encapsulates a primitive function in expression.

    Primitives are functions that take arguments and return a value.
    """

    __slots__ = ('name', 'arity', 'args', 'ret', 'seq')

    def __init__(self, name: str, args: list[type[object]], ret: object):
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret
        args_str = ", ".join(map("{{{0}}}".format, list(range(self.arity))))
        self.seq = f"{name}({args_str})"

    def format(self, *args: object) -> str:
        return self.seq.format(*args)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Primitive):
            return all(getattr(self, slot) == getattr(other, slot) for slot in self.__slots__)

        return NotImplemented

class Ephemeral(Terminal):
    """
    Class that encapsulates a terminal which value is set when the object is created.

    To mutate the value, a new object has to be generated.
    This is an abstract base class. When subclassing, a staticmethod `func` must be defined.
    """

    def __init__(self) -> None:
        super().__init__(self.func(), symbolic=False, ret=self.ret)

    @staticmethod
    def func() -> object:
        """
        Return a random value used to define the ephemeral state.
        """
        raise NotImplementedError("Ephemeral constants must define a staticmethod 'func'.")
