import pickle

from .runner import Runner

from ..policies.utils import load_policy
from ..optimizers.utils import get_best_phenotype
from ..environments.utils import load_environment


class PreprocessingDatageneratorBuilder:

    def __init__(self):
        self._instance = None

    def __call__(self, **config):

        if not self._instance:

            # Create env:
            config_env = config['env']
            env = load_environment(config_env)

            # Create policy:
            config_policy = config['policy']
            policy = load_policy(config_policy)

            # Internals
            policy_folder = config['runner']['policy_folder']
            output_folder = config['runner']['output_folder']

            self._instance = PreprocessingDatagenerator(env, policy, policy_folder, output_folder)
        return self._instance


class PreprocessingDatagenerator(Runner):

    def __init__(self, env, policy, policy_folder, output_folder):

        self.env = env
        self.policy = policy
        self.policy_folder = policy_folder
        self.output_folder = output_folder

    def run(self):

        # If learnable policy, load best params
        if self.policy_folder != None:
            phenotype = get_best_phenotype(self.policy_folder)
            self.policy.set_params(phenotype)

        results = self.run_policy_on_env(self.policy, self.env)

#        import numpy as np
#        results = np.array(results, object)


#        nb_evaluations = results[:, 0]
#        nb_evaluations =np.array([np.array(x) for x in nb_evaluations], object)

#        trajectories = results[:, 1]
#        trajectories = np.array([np.array(x) for x in trajectories], object)

#        targets = results[:, 2]
#        targets = np.array([np.array(x) for x in targets], object)

#        print(nb_evaluations[0])
#        print(trajectories[0])
#        print(targets)

#        exit()



        # Save results:
        with open(self.output_folder, "wb") as file:
            pickle.dump(results, file)
        print(f"Saved at {self.output_folder}")



#class APreprocessingDatagenerator(Runner):

#   def __init__(self, env, policy, input_path, output_path):
#        r"""
#        """

#        self.env = env
#        self.policy = policy

#        # Internals:
#        self.input_path = input_path
#        self.output_path = output_path

#    def run(self):



