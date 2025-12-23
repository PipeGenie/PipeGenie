# SPDX-License-Identifier: MIT-HUMAINS-Attribution
#
# Copyright (c) 2024 HUMAINS Research Group (University of Córdoba, Spain).
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License – HUMAINS Research Group Attribution Variant
# For full license text, see the LICENSE file in the repository root.

"""
Score and performance metrics to evaluate the models and pipelines.
"""

from ._classification import accuracy_score, balanced_accuracy_score, roc_auc_score
from ._regression import mean_squared_error, r2_score, root_mean_squared_error

__all__ = [
    'accuracy_score',
    'balanced_accuracy_score',
    'roc_auc_score',
    'mean_squared_error',
    'root_mean_squared_error',
    'r2_score',
]
