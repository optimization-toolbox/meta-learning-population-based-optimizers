import os
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np

class Task(ABC):

    def __init__(self, max_iteration, complete_tracking):

        self.complete_tracking = complete_tracking
        self.max_iteration = max_iteration

        if self.complete_tracking == True:
            self._x_trajectory = []

        self._y_trajectory = []
        self._trajectory = []
        self._targets = None

        self._nb_evaluations = 0

    def __call__(self, x):

        # Evaluate
        y = self.evaluate(x)
        self._nb_evaluations += 1

        # Update Info
        if self.complete_tracking == True:
            self._x_trajectory.append(x)
            self._y_trajectory.append(y)
        else:
            if len(self._y_trajectory) == 0:
                self._y_trajectory.append(y)
                self._trajectory.append(self._nb_evaluations)
            elif (y < self._y_trajectory[-1]) or (self._nb_evaluations == self.max_iteration):
                    self._y_trajectory.append(y)
                    self._trajectory.append(self._nb_evaluations)
        return y

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def nb_evaluations(self):
        return self._nb_evaluations

    @property
    def x_trajectory(self):
        return self._x_trajectory

    @property
    def y_trajectory(self):
        return self._y_trajectory

    @property
    def targets(self):
        if self._targets is None:

            try:
                fopt = self.f.fopt
                temp_f = deepcopy(self.f)
                self._targets = fopt + 10 ** np.linspace(2, self.targets_precision, self.targets_amount)
            except:

                import json
                path_prefix = os.path.join(os.path.dirname(__file__), "hpo_data/profet_data/targets/meta_" + str(self.f_id) + "_noiseless_targets.json")

                #print(path_prefix)
                with open(path_prefix) as f:
                    targets = np.array(json.load(f))
                targets = targets[self.i_id]

                #print(targets)
                traj = []
                curr = 1
                for t in targets:
                    if t < curr:
                        curr = t
                    traj.append(curr)
                traj = np.array(traj)
                self._targets = traj
                
                #sobol_search = []
                #for target in targets:
                #    traj = []
                #    curr = 1
                #    for t in target:
                #        if t < curr:
                #            curr = t
                #        traj.append(curr)
                #    sobol_search.append(traj)
                #sobol_search = np.array(sobol_search)
                #self._targets = sobol_search

        return self._targets

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass


class TaskBBOB(Task):

    def __init__(self, max_iteration, complete_tracking):
        super().__init__(max_iteration, complete_tracking)

    def __str__(self):
        return f'TaskBBOB => |F{self.f_id}({self.i_id}) D:{self.dim}|'

    def build(self, f_id, i_id, dim, targets_amount, targets_precision):

        # http://cma.gforge.inria.fr/apidocs-pycma/cma.bbobbenchmarks.html
        from cma import bbobbenchmarks as bn
        self.f_id = f_id
        self.i_id = i_id
        self.dim = dim
        self.targets_amount = targets_amount
        self.targets_precision = targets_precision

        self.f = eval(f'bn.F{self.f_id}({self.i_id})')
        return self

    def evaluate(self, x):
        return self.f(x)

class TaskHPO(Task):

    def __init__(self, max_iteration, complete_tracking):
        super().__init__(max_iteration, complete_tracking)

    def __str__(self):
        return f'TaskHPO => |F{self.f_id}({self.i_id}) D:{self.dim}|'

    def build(self, f_id, i_id, dim, noise=False):

        from .hpo_data.meta_benchmarks import meta_svm
        from .hpo_data.meta_benchmarks import meta_fcnet
        from .hpo_data.meta_benchmarks import meta_xgboost

        self.f_id = f_id
        self.i_id = i_id
        self.dim = dim
        self.noise = noise
        path_prefix = os.path.join(os.path.dirname(__file__), "hpo_data/profet_data/samples/" + str(self.f_id) + "/")


        path_objective = f'{path_prefix}sample_objective_{str(self.i_id)}.pkl'
        path_cost = f'{path_prefix}sample_cost_{str(self.i_id)}.pkl'

        if self.f_id == "svm":
            self.f, parameter_space = meta_svm(path_objective, path_cost, self.noise)
        elif self.f_id == "fcnet":
            self.f, parameter_space = meta_fcnet(path_objective, path_cost, self.noise)
        elif self.f_id == "xgboost":
            self.f, parameter_space = meta_xgboost(path_objective, path_cost, self.noise)
        else:
            print(str(self.f_id) + "not implemented!")

        return self

    def evaluate(self, x):
        x = np.array(x)[np.newaxis]
        mean, std = self.f(x)
        return mean[0][0]

if __name__ == "__main__":

    # Example:
    f = TaskBBOB().build(1, 1, 2, 10, -3)

    X = [[0.0, 1.0], [0.2, 0.4], [0.3, 0.4]]
    Y = [f(x) for x in X]

    print()
    print('==================================================')
    print(f'Task: {f}')
    print('==================================================')
    print('x_trajectory | y_trajectory')
    for x, y in zip(f.x_trajectory, f.y_trajectory):
        print(f'{x}   | {y}')
    print('--------------------------------------------------')
    print(f'Targets solved: {f.get_percentage_of_targets_solved()}')
    print('Targets:')
    print(f.targets)
    print()

    # Example:
    f = TaskHPO().build('svm', 1, 2, 10, -3)

    X = [[0.0, 1.0], [0.2, 0.4], [0.3, 0.4]]
    Y = [f(x) for x in X]

    print()
    print('==================================================')
    print(f'Task: {f}')
    print('==================================================')
    print('x_trajectory | y_trajectory')
    for x, y in zip(f.x_trajectory, f.y_trajectory):
        print(f'{x}   | {y}')
    print('--------------------------------------------------')
    print(f'Targets solved: {f.get_percentage_of_targets_solved()}')
    print('Targets:')
    print(f.targets)
    print()
