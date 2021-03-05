import cma
import numpy as np

from .policy import Policy


class CMAES(Policy):
    r"""Class for Deep Neural Network as Policy Model"""

    def __init__(self, config):

        super().__init__(config)

        # Check if variable are defined:
        self.dim
        self.x0_interval
        #self.popsize
        self.sigma0
        self.maxiter
        self.bounds


    def forward(self, observation, action=None, reward=None, time=None):

        if observation is not None:
            self.optimizer.tell(self.offspring, observation)

        self.offspring = self.optimizer.ask()
        self.offspring = np.array(self.offspring)
        return self.offspring

    def get_params(self):
        return

    def set_params(self, new_weights):
        return

    def reset(self, seed=None):

        gen = np.random.RandomState(seed)
        x0 = np.ones(self.dim) * gen.uniform(self.x0_interval[0], self.x0_interval[1])
        self.optimizer = cma.CMAEvolutionStrategy(x0, self.sigma0, {'verb_disp': 0, 'seed': seed, 'bounds': self.bounds, 'maxiter': self.maxiter})

        return self
