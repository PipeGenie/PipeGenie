#
# Copyright (c) 2024 University of Córdoba, Spain.
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License with Attribution Clause
# For full license text, see the LICENSE file in the repo root.
#

"""
Metrics to assess performance of classification models given predicted classes or probabilities.

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


def accuracy_score(
    y_true: 'Union[list, NDArray]',
    y_pred: 'Union[list, NDArray]',
) -> float:
    """
    Accuracy classification score.

    In multilabel classification, this function computes subset accuracy: the set of labels
    predicted for a sample must exactly match the corresponding set of labels in y_true.

    Parameters
    ----------
    y_true : 1d array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : 1d array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.

    Returns
    -------
    accuracy : float
        The fraction of correctly classified samples.

    Examples
    --------
    >>> y_true = [0, 1, 2, 3, 4]
    >>> y_pred = [0, 1, 2, 3, 4]
    >>> accuracy_score(y_true, y_pred)
    1.0
    >>> y_pred = [0, 1, 2, 3, 5]
    >>> accuracy_score(y_true, y_pred)
    0.8
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(y_true == y_pred)


def balanced_accuracy_score(
    y_true: 'Union[list, NDArray]',
    y_pred: 'Union[list, NDArray]',
) -> float:
    """
    Compute the balanced accuracy.

    The balanced accuracy in binary and multiclass classification problems to deal with
    imbalanced datasets. It is defined as the average of recall obtained on each class.

    Parameters
    ----------
    y_true : 1d array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_pred : 1d array-like of shape (n_samples,)
        Predicted labels, as returned by a classifier.

    Returns
    -------
    balanced_accuracy : float
        The balanced accuracy.

    Examples
    --------
    >>> y_true = [0, 1, 0, 0, 1, 0]
    >>> y_pred = [0, 1, 0, 0, 1, 0]
    >>> balanced_accuracy_score(y_true, y_pred)
    1.0
    >>> y_pred = [0, 1, 0, 0, 0, 1]
    >>> balanced_accuracy_score(y_true, y_pred)
    0.625
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    n_classes = classes.shape[0]
    recall = np.zeros(n_classes)

    for i, cls in enumerate(classes):
        recall[i] = np.mean(y_true[y_true == cls] == y_pred[y_true == cls])

    return np.mean(recall)


def roc_auc_score(
    y_true: 'Union[list, NDArray]',
    y_score: 'Union[list, NDArray]',
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from scores.

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------
    y_true : 1d array-like of shape (n_samples,)
        Ground truth (correct) labels.

    y_score : 1d array-like of shape (n_samples,)
        Target scores, can either be probability estimates of the positive class,
        confidence values, or non-thresholded measure of decisions.

    Returns
    -------
    roc_auc : float
        The ROC AUC score.

    Examples
    --------
    >>> y_true = [0, 1, 0, 1, 0, 0]
    >>> y_score = [0.1, 0.9, 0.2, 0.3, 0.8, 0.4]
    >>> roc_auc_score(y_true, y_score)
    0.75
    >>> y_score = [0.1, 0.9, 0.2, 0.7, 0.8, 0.4]
    >>> roc_auc_score(y_true, y_score)
    0.875
    >>> y_score = [0.1, 0.9, 0.6, 0.7, 0.2, 0.4]
    >>> roc_auc_score(y_true, y_score)
    1.0
    """
    desc_score_indices = np.argsort(y_score)[::-1]
    y_true = np.asarray(y_true)[desc_score_indices]
    y_score = np.asarray(y_score)[desc_score_indices]

    positives = np.sum(y_true)
    negatives = y_true.shape[0] - positives

    tp = 0
    fp = 0
    tpr_prev = 0
    fpr_prev = 0
    auc = 0

    for i in range(y_true.shape[0]):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1

        tpr = tp / positives
        fpr = fp / negatives

        auc += (tpr + tpr_prev) * (fpr - fpr_prev) / 2
        tpr_prev = tpr
        fpr_prev = fpr

    return auc
