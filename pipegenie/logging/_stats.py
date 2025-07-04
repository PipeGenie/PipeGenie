#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

# This file contains classes based on DEAP's tools.Statistics and
# tools.MultiStatistics to compute statistics from data.
# Source:
# - tools.Statistics: https://github.com/DEAP/deap/blob/master/deap/tools/support.py#L150
# - tools.MultiStatistics: https://github.com/DEAP/deap/blob/master/deap/tools/support.py#L209

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from typing import Union


def _as_is(obj: object) -> object:
    """
    Return directly the object as is.
    """
    return obj

class Statistics:
    """
    Compile statistics from a data using a set of functions.

    Parameters
    ----------
    key : callable, default=_as_is
        Function to extract the values from the data. By default, it returns the object as is.

    Attributes
    ----------
    key : callable
        Function to extract the values from the data.

    functions : dict of str: callable
        Dictionary of functions to compute the statistics.

    fields : list of str
        List of the names of the registered functions.

    Examples
    --------
    >>> statistics = Statistics()
    >>> statistics.register('mean', lambda x: sum(x) / len(x))
    >>> data = [1, 2, 3, 4, 5]
    >>> result = statistics.compile(data)
    >>> result['mean']
    3
    >>> statistics = Statistics(key=lambda x: x[1])
    >>> statistics.register('mean', lambda x: sum(x) / len(x))
    >>> data = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    >>> result = statistics.compile(data)
    >>> result['mean']
    3
    """

    def __init__(self, key: 'Callable[..., object]' = _as_is):
        self.key = key
        self.functions: 'dict[str, Callable[[Iterable[object]], Union[int, float]]]' = {}
        self.fields: list[str] = []

    def register(
        self,
        name: str,
        function: 'Callable[[Iterable[object]], Union[int, float]]',
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        Register a function to compute a statistic.

        Parameters
        ----------
        name : str
            Name of the statistic.

        function : callable
            Function to compute the statistic.

        args : tuple
            Additional arguments to pass automatically to the registered function.

        kwargs : dict of str: object
            Additional keyword arguments to pass automatically to the registered function.
        """
        self.functions[name] = partial(function, *args, **kwargs)
        self.fields.append(name)

    def compile(self, data: 'Iterable[object]') -> 'dict[str, Union[int, float]]':
        """
        Apply the registered functions to the data and return the results as a dictionary.

        Parameters
        ----------
        data : iterable of object
            Data to compute the statistics.

        Returns
        -------
        entry : dict of str: int/float
            Dictionary with the computed statistics.
        """
        values = tuple(self.key(item) for item in data)

        def safe_call(
            function: 'Callable[[Iterable[object]], Union[int, float]]',
            values: 'Iterable[object]',
        ) -> 'Union[int, float]':
            try:
                return function(values)
            except Exception:
                return -1

        return {
            name: safe_call(function, values)
            for name, function in self.functions.items()
        }

class MultiStatistics(dict[str, Statistics]):
    """
    Dictionary of Statistics objects to compute multiple statistics with different keys at once.

    Attributes
    ----------
    fields : list of str
        Ordered list of the names of the registered statistics.

    Examples
    --------
    >>> statistics_1 = Statistics(key=lambda x: x[0])
    >>> statistics_2 = Statistics(key=lambda x: x[1])
    >>> multi_statistics = MultiStatistics(stat_1=statistics_1, stat_2=statistics_2)
    >>> multi_statistics.register('mean', lambda x: sum(x) / len(x))
    >>> data = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5)]
    >>> result = multi_statistics.compile(data)
    >>> result['stat_1']['mean']
    1
    >>> result['stat_2']['mean']
    3
    """

    def __init__(self, **statistics: Statistics):
        super().__init__(statistics)
        self.fields = sorted(self.keys())

    def register(
        self,
        name: str,
        function: 'Callable[[Iterable[object]], Union[int, float]]',
        *args: object,
        **kwargs: object,
    ) -> None:
        """
        Register a function to compute in every statistics.

        Parameters
        ----------
        name : str
            Name of the statistic.

        function : callable
            Function to compute the statistic.

        args : tuple
            Additional arguments to pass automatically to the registered function.

        kwargs : dict of str: object
            Additional keyword arguments to pass automatically to the registered function.
        """
        for statistics in self.values():
            statistics.register(name, function, *args, **kwargs)

    def compile(self, data: 'Iterable[object]') -> 'dict[str, dict[str, Union[int, float]]]':
        """
        Apply the registered functions to the data and return the results as a dictionary.

        Parameters
        ----------
        data : iterable of object
            Data to compute the statistics.

        Returns
        -------
        record : dict of str: dict of str: int/float
            Dictionary with the computed statistics.
        """
        return {
            name: statistics.compile(data)
            for name, statistics in self.items()
        }
