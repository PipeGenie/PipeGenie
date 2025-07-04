#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

from functools import update_wrapper
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ParamSpec, TypeVar

    P = ParamSpec('P')
    T = TypeVar('T')


class _AvailableIfDescriptor:
    """
    A decorator to make a method available on an instance.

    This decorator ensures that the method is only available if
    a specified condition function returns True.
    """

    def __init__(
        self,
        func: 'Callable[P, T]',
        check_func: 'Callable[[object], bool]',
        attr_name: str,
    ):
        """
        Initialize the decorator with a condition method.

        Parameters
        ----------
        func : callable
            The method to decorate.

        check_func : callable
            The condition method that determines whether the decorated
            method should be available on the instance.

        attr_name : str
            The name of the decorated method.
        """
        self.func = func
        self.condition_method = check_func
        self.attr_name = attr_name
        update_wrapper(self, func)

    def __get__(self, instance: object, owner: type[object]) -> 'Callable[P, T]':
        """
        Return the function if the condition is met, otherwise raise an AttributeError.

        Parameters
        ----------
        instance : object
            The instance of the class where the descriptor was accessed.

        owner : type
            The class where the descriptor was accessed.

        Returns
        -------
        callable
            The decorated method if the condition is met.

        Raises
        ------
        AttributeError
            If the condition is not met.
        """
        if instance is not None:
            if self.condition_method(instance):
                result: 'Callable[P, T]' = self.func.__get__(instance, owner)
                return result

            raise AttributeError(f"'{owner.__name__}' object has no attribute '{self.attr_name}'")

        return self

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def available_if(
    check: 'Callable[[object], bool]',
) -> 'Callable[[Callable[P, T]], _AvailableIfDescriptor]':
    """
    Make an attribute available only if a specified condition method returns True.

    Parameters
    ----------
    check : callable
        When called, this function should return whether the decorated
        method should be available or not on the instance.

    Returns
    -------
    func : callable
        Callable makes the decorated method available only if the condition function returns True.
        Otherwise, the method is unavailable.
    """
    return lambda func: _AvailableIfDescriptor(func, check, attr_name=func.__name__)
