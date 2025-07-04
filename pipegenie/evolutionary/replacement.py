#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Population replacement strategies.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipegenie.evolutionary._individual import Individual


class ReplacementBase(ABC):
    """
    Base class for replacement operators.
    """

    @abstractmethod
    def _perform_replacement(
        self,
        population: list['Individual'],
        offspring: list['Individual'],
        elite: list['Individual'],
    ) -> list['Individual']:
        """
        Perform the replacement of individuals in the population.
        """
        raise NotImplementedError("Method '_perform_replacement' must be implemented in subclass")

    def replace(
        self,
        population: list['Individual'],
        offspring: list['Individual'],
        elite: list['Individual'],
    ) -> list['Individual']:
        """
        Create the new population using the current population, the offspring and the elite.

        Parameters
        ----------
        population : list of Individual
            The current population

        offspring : list of Individual
            The offspring created in the generation

        elite : list of Individual
            The elite individuals

        Returns
        -------
        new_population : list of Individual
            The new population
        """
        return self._perform_replacement(population, offspring, elite)


class ElitistGenerationalReplacement(ReplacementBase):
    """
    Generational replacement with elitism.

    The population is completely replaced by the offspring.
    Then, the elite is inserted replacing the worst individuals.

    Parameters
    ----------
    elitism : int, default=1
        The number of elite individuals to insert in the population
    """

    def __init__(self, elitism: int = 1):
        self.elitism = elitism

    def _perform_replacement(
        self,
        population: list['Individual'],
        offspring: list['Individual'],
        elite: list['Individual'],
    ) -> list['Individual']:
        """
        Perform the replacement of individuals in the population.
        """
        new_pop = sorted(offspring, key=lambda ind: ind.fitness.valid)
        idx = next((i for i, ind in enumerate(new_pop) if ind.fitness.valid), 0)
        new_pop[idx:] = sorted(new_pop[idx:], key=lambda ind: ind.fitness.weighted_value)
        elitism = min(self.elitism, len(elite))  # Ensure elitism is not greater than elite size
        elite_to_insert = [ind for ind in elite if ind not in new_pop][:elitism]
        new_pop[:len(elite_to_insert)] = elite_to_insert
        return new_pop

    def __str__(self) -> str:
        return f"ElitistGenerationalReplacement(elitism={self.elitism})"

