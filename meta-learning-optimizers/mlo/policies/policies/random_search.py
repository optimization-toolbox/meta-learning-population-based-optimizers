import cma
import numpy as np

from .policy import Policy


class BatchRandomSearch(Policy):
    r"""Class for Deep Neural Network as Policy Model"""

    def __init__(self, config):

        super().__init__(config)

        # population_size
        # bounds
        # dim
        # global_seed

    def forward(self, observation=None, action=None, reward=None, time=None):
        # def forward(self, observation=None):

        solutions = self.generator.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

        return solutions

    def get_params(self):
        return

    def set_params(self, new_weights):
        return

    def reset(self, seed=None):

        self.generator = np.random.RandomState(seed)

        return self
