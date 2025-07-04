#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Metrics to assess performance of regression models given predicted values.

Functions named as ``*_score`` return a scalar value to maximize:
the higher the better.

Functions named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import Union

    from numpy.typing import NDArray


def mean_squared_error(
    y_true: 'Union[list, NDArray]',
    y_pred: 'Union[list, NDArray]',
) -> float:
    """
    Mean squared regression error.

    Parameters
    ----------
    y_true : 1d array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : 1d array-like of shape (n_samples,)
        Predicted values, as returned by a regressor.

    Returns
    -------
    error : float
        The error commited by the regressor.

    Examples
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> mean_squared_error(y_true, y_pred)
    0.375
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2, axis=0)


def root_mean_squared_error(
    y_true: 'Union[list, NDArray]',
    y_pred: 'Union[list, NDArray]',
) -> float:
    """
    Root mean squared regression error.

    Parameters
    ----------
    y_true : 1d array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : 1d array-like of shape (n_samples,)
        Predicted values, as returned by a regressor.

    Returns
    -------
    error : float
        The error commited by the regressor.

    Examples
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> root_mean_squared_error(y_true, y_pred)
    0.6123724356957945
    """
    return mean_squared_error(y_true, y_pred) ** (1 / 2)


def r2_score(
    y_true: 'Union[list, NDArray]',
    y_pred: 'Union[list, NDArray]',
) -> float:
    """
    R^2 (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).

    Parameters
    ----------
    y_true : 1d array-like of shape (n_samples,)
        Ground truth (correct) values.

    y_pred : 1d array-like of shape (n_samples,)
        Predicted values, as returned by a regressor.

    Returns
    -------
    score : float
        The R^2 score of the regressor.

    Examples
    --------
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)
    0.9486081370449679
    >>> y_pred = [4, 4, 4, 4]
    >>> r2_score(y_true, y_pred)
    -0.17344753747323338
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerator / denominator)
