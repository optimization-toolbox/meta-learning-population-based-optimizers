r"""
Abstract base module for all PI4py policies.
"""

from abc import ABC, abstractmethod
import torch.nn as nn


class Policy(ABC, nn.Module):
    r"""Abstract base class for policies."""

    def __init__(self, config):
        r"""Constructor for the Policy base class.

            Args:
                config: configurable setting for the optimizer

        """
        super().__init__()
        self.config = config
        self.__dict__.update(config)

    @abstractmethod
    def forward(self):
        r"""Prediction method."""
        pass

    @abstractmethod
    def set_params(self, new_params):
        r"""Set the params of the policy as new_params."""
        pass

    @abstractmethod
    def get_params(self):
        r"""Get params of the policy."""
        pass

    @abstractmethod
    def reset(self, seed=None):
        r"""Initialize policy with seed."""
        pass
