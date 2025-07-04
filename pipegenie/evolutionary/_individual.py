#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Individual and Fitness classes for the Genetic Programming algorithm.
"""

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from pipegenie.syntax_tree._encoding import SyntaxTreePipeline

if TYPE_CHECKING:
    from typing import Optional, Union

    from numpy.typing import NDArray

    from pipegenie.syntax_tree._encoding import NonTerminalNode, TerminalNode


class Fitness:
    """
    Class representing the fitness of an individual.

    Parameters
    ----------
    weights : tuple of floats
        The weights associated with the fitness values.
    """

    __slots__ = ('_values', 'weights')

    def __init__(self, weights: tuple[float, ...]):
        if weights is None or len(weights) == 0:
            raise ValueError("Weights cannot be None or empty")

        self.weights = weights
        self._values: tuple[float, ...] = ()

    @property
    def values(self) -> tuple[float, ...]:
        return self._values

    @values.setter
    def values(self, values: tuple[float, ...]) -> None:
        if values is None or len(values) == 0:
            raise ValueError("Values cannot be None or empty")
        if len(values) != len(self.weights):
            raise ValueError("Values and weights must have the same length")

        self._values = values

    @values.deleter
    def values(self) -> None:
        self._values = ()

    @property
    def valid(self) -> bool:
        """
        Check if the fitness value is valid.

        Returns
        -------
        valid : bool
            True if the fitness value is valid, False otherwise.
        """
        return (
            self._values is not None and
            len(self.values) > 0 and
            not np.isnan(self.weighted_value)
        )

    @property
    def weighted_value(self) -> float:
        """
        Calculate the weighted value of the fitness.

        Returns
        -------
        weighted_value : float
            The weighted value of the fitness.
        """
        if self._values is None:
            raise ValueError("Fitness values have not been set")

        return sum(v * w for v, w in zip(self.values, self.weights, strict=True))

    def invalidate(self) -> None:
        """
        Invalidate the fitness values.
        """
        self._values = ()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fitness):
            raise ValueError(f"Cannot compare Fitness with {type(other)}")

        return self.values == other.values and self.weights == other.weights


class Individual(SyntaxTreePipeline):
    """
    Class representing an individual in the population.

    Contains the machine learning pipeline and its associated fitness.

    Parameters
    ----------
    content : list of TerminalNode and NonTerminalNode objects
        List of nodes representing the pipeline.

    fitness : Fitness, default=None
        The fitness of the individual.
        If None, a default fitness object is created with a single weight of 1.0.
    """

    def __init__(
        self,
        content: list['Union[TerminalNode, NonTerminalNode]'],
        fitness: 'Optional[Fitness]' = None,
    ):
        super().__init__(content)
        self.content = content
        self.fitness = fitness if fitness is not None else Fitness((1.0,))
        self.diversity: float = np.nan
        self.prediction: 'NDArray' = np.array([])

    def clone(self) -> 'Individual':
        """
        Clone the individual.

        Returns
        -------
        Individual
            A copy of the individual.
        """
        return deepcopy(self)

    def reset(self) -> None:
        """
        Reset the individual.
        """
        super().reset()
        self.fitness.invalidate()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            raise ValueError(f"Cannot compare Individual with {type(other)}")

        return super().__eq__(other) and self.fitness == other.fitness
