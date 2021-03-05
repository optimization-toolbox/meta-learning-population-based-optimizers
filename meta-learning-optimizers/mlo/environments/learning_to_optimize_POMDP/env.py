import os

import numpy as np

from .task_generator import TaskGenerator
from .render import RenderView


class LearningToOptimizePOMDP:

    r'''
        Example configuration:

        env:
          id: GYM4bbo:MetaEvolution-v1
          task_config:
            name: bbob
            f_ids:
            - 5
            i_ids:
            - 1
            - 3
            dim: 2
            targets:
                amount: 51
                precision: -3
            repeat_task: 1
            mode: deterministic
          env_options:
            max_iteration: 30
            reward_function: percentage_of_targets_solved
            save_results: true
          internals:
            debug: true

    '''

    def __init__(self, config):

        # Env Options:
        self.plot_path = config['env_options']['plot_path']
        self.debug = config['env_options']['debug']

        # Internals:
        self.results = []
        self.complete_tracking = True if self.plot_path != None else False
        self.render_viewer = None
        self.max_iteration = config['task_config']['max_iteration']

        # Task Generator:
        self.task_generator = TaskGenerator(config['task_config'], self.complete_tracking)

    def reset(self):
        r'''
            gym method to reset the environment
        '''

        # Sample task:
        self.f = next(self.task_generator)
        self.time = 0.0

        # Create plot folder:
        if self.plot_path != None:
            try:
                os.makedirs(self.plot_path)
            except:
                pass
            if self.render_viewer == None:
                self.render_viewer = RenderView(self.plot_path).build(self.f).save()
            else:
                self.render_viewer.reset()
                self.render_viewer.build(self.f).save()

        # Return Initial Observation
        return None

    def step(self, action):
        r'''
            gym method to step the environment
        '''

        # Dynamics:
        self.action = action
        self.observation = np.array([self.f(x) for x in action])

        # Done:
        done = True if self.f.nb_evaluations >= self.max_iteration else False
        if done == True:
            self.update_results()

#        if self.debug:
#            print(f"\n\n========== GENERATION: {self.generation} ==========\n")
#            print(f"OBSERVATION {self.generation}\n{self.observation}")
#            print(f"ACTION {self.generation}\n{self.action}")
#            print("====================================================\n\n")

        # Get info:
        self.time += len(self.action)
        time = self.time / self.max_iteration

        return self.observation, None, done, None

    def render(self):
        r'''
            gym method to render the environment
        '''

        if self.plot_path != None:
            self.render_viewer.add(self.action).save()

    def update_results(self):
        
        if self.f.nb_evaluations != 0:
            if self.complete_tracking:
                result = [self.f.trajectory, self.f.y_trajectory, self.f.x_trajectory, self.f.targets]
            else:
                result = [self.f.trajectory, self.f.y_trajectory, self.f.targets]
            self.results.append(result)

    def close(self):
        r'''
            gym method to close the environment
        '''
        return self.results

    def clean(self):
        del self.results
        del self.f
        self.results = []

    def __len__(self):
        return len(self.task_generator)
