#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Utility functions.
"""

def is_number(string: str) -> bool:
    """
    Check if a string is a number.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    is_number : bool
        True if the string is a number, False otherwise.
    """
    try:
        float(string)
        return True
    except ValueError:
        return False

def is_bool(string: str) -> bool:
    """
    Check if a string is a boolean.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    is_bool : bool
        True if the string is a boolean, False otherwise.
    """
    return string.lower() in ('true', 'false')

def is_tuple(string: str) -> bool:
    """
    Check if a string is a tuple.

    Parameters
    ----------
    string : str
        The string to check.

    Returns
    -------
    is_tuple : bool
        True if the string is a tuple, False otherwise.
    """
    return string.startswith('(') and string.endswith(')')
