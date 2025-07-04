#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Parents mating strategies.
"""

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipegenie.evolutionary._individual import Individual


class SelectionBase(ABC):
    """
    Base class for selection operators.
    """

    @abstractmethod
    def _perform_selection(self, population: list['Individual'], k: int = 1) -> list['Individual']:
        """
        Perform the selection of individuals from the population.
        """
        raise NotImplementedError("Method '_perform_selection' must be implemented in subclass")

    def select(self, population: list['Individual'], k: int = 1) -> list['Individual']:
        """
        Select individuals from the population.

        Parameters
        ----------
        population : list of Individual
            The population to select from

        k : int, default=1
            The number of individuals to select from the population

        Returns
        -------
        parents : list of Individual
            The selected individuals
        """
        return self._perform_selection(population, k)


class RandomSelection(SelectionBase):
    """
    Random selection operator.

    Selects individuals from the population at random.
    """

    def _perform_selection(self, population: list['Individual'], k: int = 1) -> list['Individual']:
        """
        Perform the selection of individuals from the population.
        """
        return random.choices(population, k=k)

    def __str__(self) -> str:
        return "RandomSelection"

class TournamentSelection(SelectionBase):
    """
    Tournament selection operator.

    Selects the best individual considering the weighted fitness value
    from a random subset of the population.

    Parameters
    ----------
    tournament_size : int, default=3
        The size of the random subset of the population to select from
    """

    def __init__(self, tournament_size: int = 3):
        self.tournament_size = tournament_size

    def _perform_selection(self, population: list['Individual'], k: int = 1) -> list['Individual']:
        """
        Perform the selection of individuals from the population.
        """
        selected = []

        for _ in range(k):
            tournament = random.choices(population, k=self.tournament_size)
            selected.append(max(tournament, key=lambda x: x.fitness.weighted_value))

        return selected

    def __str__(self) -> str:
        return f"TournamentSelection(tournament_size={self.tournament_size})"
