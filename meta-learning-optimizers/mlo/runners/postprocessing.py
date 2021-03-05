import pickle

import numpy as np
import matplotlib.pyplot as plt

from .runner import Runner
from .utils import get_rt_table, get_simulated_rt_table, show_ecdf_graph


from .runner import Runner

class PostprocessingMetalossesBuilder:

    def __init__(self):
        self._instance = None

    def __call__(self, **config):
        if not self._instance:

            title = config['runner']['title']
            train_paths = config['runner']['train_paths']
            val_paths = config['runner']['val_paths']
            policy_data = config['runner']['policy_data']
            names = config['runner']['names']

            self._instance = PostprocessingMetalosses(title, train_paths, val_paths, policy_data, names)
        return self._instance

class PostprocessingMetalosses(Runner):

    def __init__(self, title, train_paths, val_paths, policy_data, names):

        self.title = title
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.policy_data = policy_data
        self.names = names

        self.colors = ['tab:blue', 'tab:green', 'tab:red']
        self.linestyles = ['-', '--', ':']

    def plot_policy(self):

        # Load Results:
        with open(self.policy_data, "rb") as file:
            train_costs, validation_costs = pickle.load(file)

        # Plot Train and Validation:
        self._x = list(range(len(train_costs)))
        self.ax1.plot(self._x, train_costs, color=self.colors[0], linestyle=self.linestyles[0], label='LPBO')
        self.ax2.plot(self._x, validation_costs, color=self.colors[0], linestyle=self.linestyles[0], label='LPBO')

    def plot_alg(self, i, name, train_path, val_path):

        # Load results:
        with open(train_path, "rb") as file:
            train_results = pickle.load(file)

        with open(val_path, "rb") as file:
            val_results = pickle.load(file)
        
        # Calculate Cost:
        train_cost = self.calculate_cost(train_results)
        val_cost = self.calculate_cost(val_results)

        # Plot Data:
        self.ax1.hlines(train_cost, 0, len(self._x), color=self.colors[i], linestyle=self.linestyles[i], label=name)
        self.ax2.hlines(val_cost, 0, len(self._x), color=self.colors[i], linestyle=self.linestyles[i], label=name)


    def run(self):

        # Create Figure (train and validation):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)

        # Get train data:
        self.plot_policy()

        for i, (name, train_path, val_path) in enumerate(zip(self.names, self.train_paths, self.val_paths)):

            self.plot_alg(i+1, name, train_path, val_path)

        self.ax1.grid(True, linewidth=0.1)
        self.ax2.grid(True, linewidth=0.1)
        self.fig.suptitle(f'Meta-Losses: {self.title}')
        self.ax1.set_title("Meta-Train Losses", fontsize=12)
        self.ax2.set_title("Meta-Validation Losses", fontsize=12)
        self.ax1.set_xlabel("Training Generations")
        self.ax2.set_xlabel("Training Generations")
        for ax in [self.ax1, self.ax2]:
            ax.legend()
        self.fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
        self.fig.text(0.05, 0.5, 'Avg RT to reach function+target pairs', ha='center', va='center', rotation='vertical', fontsize=12)
        plt.savefig("metaloss.pdf", format='pdf')

class PostprocessingECDFSBuilder:

    def __init__(self):
        self._instance = None

    def __call__(self, **config):
        if not self._instance:

            title = config['runner']['title']
            policy_data = config['runner']['policy_data']
            cmaes_data = config['runner']['cmaes_data']
            randomsearch_data = config['runner']['randomsearch_data']
            names = config['runner']['names']

            self._instance = PostprocessingECDFS(title, policy_data, cmaes_data, randomsearch_data, names)
        return self._instance

class PostprocessingECDFS(Runner):

    def __init__(self, title, policy_data, cmaes_data, randomsearch_data, names):

        self.title = title
        self.policy_data = policy_data
        self.cmaes_data = cmaes_data
        self.randomsearch_data = randomsearch_data
        self.names = names

        self.colors = ['tab:blue', 'tab:green', 'tab:red']
        self.linestyles = ['-', '--', ':']

    def run(self):


        # Policy Results:
        with open(self.policy_data, "rb") as file:
            policy_results = pickle.load(file)

        # CMAES Results:
        with open(self.cmaes_data, "rb") as file:
            cmaes_results = pickle.load(file)

        # RandomSearch Results:
        with open(self.randomsearch_data, "rb") as file:
            randomsearch_results = pickle.load(file)

        # Show
        self.show(policy_results, cmaes_results, randomsearch_results)

    def show(self, policy_results, cmaes_results, randomsearch_results):

        randomsearch_ecdf = self.results_to_ecdf(randomsearch_results)
        cmaes_ecdf = self.results_to_ecdf(cmaes_results)
        policy_ecdf = self.results_to_ecdf(policy_results)

        ecdfs = [policy_ecdf, cmaes_ecdf, randomsearch_ecdf]
        show_ecdf_graph(ecdfs, ['LPBO', 'CMA-ES', 'Random Search'], title=self.title)

    def results_to_ecdf(self, results):


        results = np.array(results, object)


        nb_evaluations = results[:, 0]
        nb_evaluations =np.array([np.array(x) for x in nb_evaluations], object)

        trajectories = results[:, 1]
        trajectories = np.array([np.array(x) for x in trajectories], object)
        
        targets = results[:, 2]
        targets = np.array([np.array(x) for x in targets], object)

        rt_table = get_rt_table(nb_evaluations, trajectories, targets)

        max_evaluations = results[0][0][-1]
        dim = 2
        #print(max_evaluations)
        #exit()

        simulated_rt_table = get_simulated_rt_table(rt_table, 1000, max_evaluations, dim)


        x = np.sort(simulated_rt_table.flatten()) / dim
        #x = np.sort(rt_table.flatten())
        y = np.arange(len(x))/float(len(x))

        return [x, y]