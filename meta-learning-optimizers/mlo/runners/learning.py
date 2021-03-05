import pickle
import numpy as np

from .runner import Runner

from ..policies.utils import load_policy
from ..optimizers.utils import load_optimizer
from ..optimizers.utils import load_mpidata, get_best_phenotype_generator
from ..environments.utils import load_environment


class LearningTrainBuilder:

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

            # Create optimizer:
            config_optimizer = config['optimizer']
            config_optimizer['initial_guess'] = policy.get_params()
            optimizer = load_optimizer(config_optimizer)
            config_optimizer.pop('initial_guess')

            # Internals
            steps = config['runner']['steps']
            global_seed = config['runner']['global_seed']

            self._instance = LearningTrain(env, policy, optimizer, steps, global_seed)
        return self._instance

class LearningValidationBuilder:

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

            policy_folder = config['runner']['policy_folder']
            output_folder = config['runner']['output_folder']

            self._instance =  LearningValidation(env, policy, policy_folder, output_folder)
        return self._instance


class LearningTrain(Runner):

    def __init__(self, env, policy, optimizer, steps, global_seed):

        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.steps = steps
        #self.gen_task = np.random.RandomState(global_seed)

    def loss(self, x):
        r''' Calculate Loss''' 

        # Set Policy:
        self.policy.set_params(x)

        # Set Env:
        self.env.reset()

        # Run Policy:
        results = self.run_policy_on_env(self.policy, self.env)

        # Get Final Cost
        return self.calculate_cost(results)

    def run(self):

        # Run Optimizer:
        self.optimizer.run(self.loss, self.steps)

class LearningValidation(Runner):

    def __init__(self, env, policy, policy_folder, output_folder):

        self.env = env
        self.policy = policy
        self.policy_folder = policy_folder
        self.output_folder = output_folder

    def run(self):

        # Get Train Costs:
        train_costs  = load_mpidata('costs', self.policy_folder).astype(float)
        train_costs = np.min(train_costs, axis=1)

        # Get Validation Costs:
        print("Running validation...")
        validation_costs = []
        best_phenotypes = get_best_phenotype_generator(self.policy_folder)
        for i, phenotype in enumerate(best_phenotypes):
            print(i)
            self.policy.set_params(phenotype)
            results = self.run_policy_on_env(self.policy, self.env)
            cost = self.calculate_cost(results)
            validation_costs.append(cost)

        # Save Train and Validation Costs:
        with open(self.output_folder, "wb") as file:
            pickle.dump([train_costs, validation_costs], file)
        print(f"Saved at {self.output_folder}")


