import numpy as np

from abc import ABCMeta, abstractmethod

from .utils import get_rt_table

class Runner(metaclass=ABCMeta):

    @abstractmethod
    def run(self):
        pass

    @staticmethod
    def run_policy_on_env(policy, env):
        r'''Run Policy on Environment'''

        for i in range(len(env)):

            # Reset policy:
            policy.reset(seed=i+1)
            observation = env.reset()

            # Loop task:
            action = None
            while True:

                # Transition:
                action = policy(observation, action, reward=None, time=None)
                observation, _, done, _ = env.step(action)

                # Render:
                env.render()

                # Finish task
                if done:
                    break
            results = env.close()

        # Close and Clean env
        results = env.close()
        env.clean()

        # Return results
        return results

    @staticmethod
    def calculate_cost(results):
        r'''Calculate final cost based on results'''

        results = np.array(results, object)

        nb_evaluations = results[:, 0]
        nb_evaluations =np.array([np.array(x) for x in nb_evaluations], object)

        trajectories = results[:, 1]
        trajectories = np.array([np.array(x) for x in trajectories], object)
        
        targets = results[:, 2]
        targets = np.array([np.array(x) for x in targets], object)

        rt_table = get_rt_table(nb_evaluations, trajectories, targets)

        #trajectories = results[:, 1]
        #trajectories = np.array([np.array(x) for x in trajectories], object)
        
        #targets = results[:, 2]
        #targets = np.array([np.array(x) for x in targets])

        #rt_table = get_rt_table(trajectories, targets)

        # Calculate Expected Runtime:
        max_evaluations = results[0][0][-1]
        nb_runs = rt_table.size
        nb_succ = len(rt_table[np.isfinite(rt_table)])
        total_eval_succ = np.sum(rt_table[np.isfinite(rt_table)])
        p_succ = nb_succ / nb_runs

        if nb_succ != 0:
            expected_runtime = ((1 - p_succ)/ p_succ) * max_evaluations + (total_eval_succ/nb_succ)
        else: 
            expected_runtime = np.inf
        return expected_runtime