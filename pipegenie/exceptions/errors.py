# SPDX-License-Identifier: MIT-HUMAINS-Attribution
#
# Copyright (c) 2024 HUMAINS Research Group (University of Córdoba, Spain).
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License – HUMAINS Research Group Attribution Variant
# For full license text, see the LICENSE file in the repository root.

import warnings

class PipegenieError(Exception):
    """Base class for all custom exceptions in my_package."""
    pass

class NoValidPipelines(PipegenieError):
    """Raised when there is not a valid pipeline in the population."""
    def __init__(self, message: str, invalid_value=None):
        self.invalid_value = invalid_value
        super().__init__(f"{message}. Invalid value: {invalid_value!r}")

class NoFittedModelError(PipegenieError):
    """Raised when the model was not fitted."""
    def __init__(self, message: str, config_key: str = None):
        self.config_key = config_key
        super().__init__(f"{message}" + (f" [key: {config_key}]" if config_key else ""))



class PipegenieWarning(Warning):
    """Base class for all package-related warnings."""
    pass

class NoValidPipelineWarning(PipegenieWarning):
    """Warns when there is not a single valid pipeline in a generation"""
    pass


def warn(message, category=PipegenieWarning, stacklevel=2):
    """Issue a package-specific warning."""
    warnings.warn(message, category=category, stacklevel=stacklevel)