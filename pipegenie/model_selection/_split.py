#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

from abc import ABC, abstractmethod
from math import ceil
from typing import TYPE_CHECKING

import numpy as np
from numpy import random

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Optional, Union

    from numpy.typing import ArrayLike, NDArray


class BaseCrossValidator(ABC):
    r"""
    Base class for all cross-validators.

    All cross-validators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword arguments
    (no \*args or \*\*kwargs).

    This is a minimal interface that all cross-validators should implement.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness when shuffling the data. Pass an int for reproducible output
        across multiple function calls.
    """

    MIN_N_SPLITS = 2

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: 'Optional[Union[int, random.RandomState]]' = None,
    ):
        if n_splits < self.MIN_N_SPLITS:
            raise ValueError("The 'n_splits' parameter should be at least 2")

        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self._rng = random.RandomState(random_state)

    @abstractmethod
    def split(
        self,
        X: 'ArrayLike',
        y: 'Optional[ArrayLike]' = None,
    ) -> 'Iterator[tuple[NDArray, NDArray]]':
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        raise NotImplementedError("split method should be implemented by the subclass")

    def set_random_state(self, random_state: 'Optional[Union[int, random.RandomState]]') -> None:
        """
        Set the random state.

        Parameters
        ----------
        random_state : int, RandomState instance or None
            Controls the randomness of the training data. Pass an int for reproducible output
            across multiple function calls.
        """
        self.random_state = random_state
        self._rng = random.RandomState(random_state) if self.shuffle else None

class KFold(BaseCrossValidator):
    """
    K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness when shuffling the data. Pass an int for reproducible output
        across multiple function calls.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: 'Optional[Union[int, random.RandomState]]' = None,
    ):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(
        self,
        X: 'ArrayLike',
        y: 'Optional[ArrayLike]' = None,
    ) -> 'Iterator[tuple[NDArray, NDArray]]':
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X = np.asarray(X)

        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        if self.shuffle:
            self._rng.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        current = 0

        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start : stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

    def __str__(self) -> str:
        return (
            f"KFold(n_splits={self.n_splits}, "
            f"shuffle={self.shuffle}, "
            f"random_state={self.random_state})"
        )

class StratifiedKFold(BaseCrossValidator):
    """
    Stratified K-Folds cross-validator.

    Provides train/test indices to split data in train/test sets. This cross-validation
    object is a variation of KFold that returns stratified folds. The folds are made by
    preserving the percentage of samples for each class.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

    shuffle : bool, default=False
        Whether to shuffle the data before splitting into batches.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness when shuffling the data. Pass an int for reproducible output
        across multiple function calls.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = False,
        random_state: 'Optional[Union[int, random.RandomState]]' = None,
    ):
        super().__init__(n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X: 'ArrayLike', y: 'ArrayLike') -> 'Iterator[tuple[NDArray, NDArray]]':
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X = np.asarray(X)
        y = np.asarray(y) if y is not None else None

        if y is None:
            raise ValueError("The 'y' parameter should not be None")
        if y.shape[0] != X.shape[0]:
            raise ValueError("The 'X' and 'y' parameters should have the same number of samples")

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, class_perm = np.unique(y_idx, return_inverse=True)
        y_encoded = class_perm[y_inv]

        n_classes = len(y_idx)
        y_counts = np.bincount(y_encoded)

        if np.all(self.n_splits > y_counts):
            raise ValueError(
                "The 'n_splits' parameter should be less "
                "than the number of samples in each class",
            )

        y_order = np.sort(y_encoded)
        allocation = np.asarray(
            [
                np.bincount(y_order[i :: self.n_splits], minlength=n_classes)
                for i in range(self.n_splits)
            ],
        )

        test_folds = np.empty(len(y), dtype=int)

        for i in range(n_classes):
            folds_for_class = np.arange(self.n_splits).repeat(allocation[:, i])

            if self.shuffle:
                self._rng.shuffle(folds_for_class)

            test_folds[y_encoded == i] = folds_for_class

        for i in range(self.n_splits):
            test_indices = np.where(test_folds == i)[0]
            train_indices = np.where(test_folds != i)[0]
            yield train_indices, test_indices

    def __str__(self) -> str:
        return (
            f"StratifiedKFold(n_splits={self.n_splits}, "
            f"shuffle={self.shuffle}, "
            f"random_state={self.random_state})"
        )


def train_test_split(
    X: 'ArrayLike',
    y: 'ArrayLike',
    test_size: float = 0.25,
    *,
    stratify: 'Optional[ArrayLike]' = None,
    shuffle: bool = True,
    random_state: 'Optional[Union[int, random.RandomState]]' = None,
) -> 'tuple[NDArray, NDArray, NDArray, NDArray]':
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,)
        The target variable for supervised learning problems.

    test_size : float, default=0.25
        Portion of the data to set aside for testing. Should be between 0.0 and 1.0.

    stratify : array-like of shape (n_samples,), default=None
        If not None, data is split in a stratified fashion, using this as the class labels.

    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the training data. Pass an int for reproducible output
        across multiple function calls.

    Returns
    -------
    X_train : ndarray of shape (n_train_samples, n_features)
        The training set.

    X_test : ndarray of shape (n_test_samples, n_features)
        The testing set.

    y_train : ndarray of shape (n_train_samples,)
        The target training set.

    y_test : ndarray of shape (n_test_samples,)
        The target testing set.
    """
    min_samples_class_stratify = 2

    X = np.asarray(X)
    y = np.asarray(y)

    # Ensure y is unidimensional
    if y.ndim > 1:
        if y.shape[1] == 1:
            y = y.ravel()
        else:
            raise ValueError("The 'y' parameter should be unidimensional")

    stratify = np.asarray(stratify) if stratify is not None else None

    if X.shape[0] != y.shape[0]:
        raise ValueError("The 'X' and 'y' parameters should have the same number of samples")

    if stratify is not None and y.shape[0] != stratify.shape[0]:
        raise ValueError("The 'y' and 'stratify' parameters should have the same length")

    if test_size <= 0 or test_size >= 1:
        raise ValueError("The 'test_size' parameter should be between 0.0 and 1.0")

    n_samples = X.shape[0]
    n_test = ceil(n_samples * test_size)
    n_train = n_samples - n_test

    rng = random.RandomState(random_state)

    if stratify is not None:
        stratify = np.asarray(stratify)
        classes, y_indices = np.unique(stratify, return_inverse=True)
        n_classes = len(classes)

        class_counts = np.bincount(y_indices)

        if np.min(class_counts) < min_samples_class_stratify:
            raise ValueError("There should be at least 2 samples per class "
                             "to perform stratified splitting.")

        test_counts = np.maximum(1, np.round((class_counts * test_size)).astype(int))
        train_counts = class_counts - test_counts
        train_indices = []

        for c in range(n_classes):
            c_indices = np.where(y_indices == c)[0]

            if shuffle:
                # Shuffle the indices of the current class
                rng.shuffle(c_indices)

            train_indices.extend(c_indices[:train_counts[c]])

        test_indices = np.setdiff1d(np.arange(n_samples), train_indices)

        if shuffle:
            # Shuffle again to mix the classes in case they were sorted
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)
    else:
        indices = np.arange(n_samples)

        if shuffle:
            rng.shuffle(indices)

        train_indices = indices[:n_train]
        test_indices = indices[n_train:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
