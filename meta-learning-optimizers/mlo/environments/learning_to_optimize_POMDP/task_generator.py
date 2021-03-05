import numpy as np

from .task import TaskBBOB, TaskHPO 
from .utils import RandomGenerator, DeterministicGenerator

#from GYM4bbo.envs.MetaEvolution.task import TaskBBOB, TaskHPO
#from GYM4bbo.envs.MetaEvolution.utils import RandomGenerator, DeterministicGenerator

    
class TaskGenerator:

    r'''
    Example task_config:

    task_task_config:
        name: bbo
        f_ids:
        - 5
        i_ids:
        - 1
        - 3
        dim: 2
        targets:
            amount: 51
            precision: -3
        repeat: 1
        mode: deterministic
    '''

    def __init__(self, task_config, complete_tracking):

        self.task_config = task_config
        self.complete_tracking = complete_tracking

        # Task Config:
        self.name = task_config['name']
        self.mode = task_config['mode']
        self.repeat = task_config['repeat']
        self.f_ids = task_config['f_ids']
        self.i_ids = task_config['i_ids']
        self.dim = task_config['dim']
        self.max_iteration = task_config['max_iteration']

        targets = task_config.get('targets')
        if targets != None:
            self.targets_amount = targets.get('amount')
            self.targets_precision = targets.get('precision')

        if self.mode == 'random':
            self.seed = task_config.get('seed')
            self.batch_size = task_config.get('batch_size')

        # Define generator:
        if self.mode == 'deterministic':
            self.generator = DeterministicGenerator(self.f_ids, self.i_ids)
        elif self.mode == 'random':
            self.generator = RandomGenerator(self.f_ids, self.i_ids, self.seed)

        # Internals:
        self.n = self.repeat

    def __next__(self):

        # Get next task:
        if self.mode == 'deterministic':
            self.f_type, self.f_idx = next(self.generator)
        elif self.mode == 'random':
            if self.n == self.repeat:
                self.f_type, self.f_idx = next(self.generator)
                self.n = 1
            else:
                self.n += 1

        # Create next task:
        if self.name == 'bbob':
            task = TaskBBOB(self.task_config['max_iteration'], self.complete_tracking)
            task.build(self.f_type, self.f_idx, self.dim, self.targets_amount, self.targets_precision)
            return task
        if self.name == 'hpo':
            task = TaskHPO(self.task_config['max_iteration'], self.complete_tracking)
            task.build(self.f_type, self.f_idx, self.dim)
            return task
        else:
            raise NameError('Task name not implemented!')

    def __iter__(self):
        return self

    def __len__(self):
        if self.mode == 'deterministic':
            length = len(self.f_ids) * (self.i_ids[1] - self.i_ids[0] + 1)
            return int(self.repeat * length)
        elif self.mode == 'random':
            return self.repeat * self.batch_size

if __name__ == "__main__":

    print("!")
    import GYM4bbo.envs.MetaEvolution

    #from GYM4bbo.envs.MetaEvolution.utils import RandomGenerator, DeterministicGenerator

    exit()

    if False:
        task_config = {'name': 'bbob',
                  'mode': 'random',
                  'f_ids': [5],
                  'i_ids': [1, 1000],
                  'dim': 2,
                  'targets': {'amount': 51, 'precision': -3},
                  'repeat': 3,
                  'batch_size': 10,
                  'seed': 1982371} 
    elif False:
        task_config = {'name': 'bbob',
                  'mode': 'deterministic',
                  'f_ids': [1],
                  'i_ids': [1, 5],
                  'dim': 2,
                  'targets': {'amount': 51, 'precision': -3},
                  'repeat': 3}
    elif True:
        task_config = {'name': 'hpo',
                  'mode': 'deterministic',
                  'f_ids': ['svm'],
                  'i_ids': [1, 4],
                  'dim': 2,
                  'targets': {'amount': 51, 'precision': -3},
                  'repeat': 1}

    bbo_generator = TaskGenerator(task_config)

    print()
    print('==================================================')
    print('Testing Task Generators...')
    print('==================================================')
    print('task_configuration:')
    print('--------------------------------------------------')
    for k, v in task_config.items():
        print(f'{k}: {v}')
    print('--------------------------------------------------')
    print(f'Generator Lenght: {len(bbo_generator)}')
    for _ in range(len(bbo_generator)):
        task = next(bbo_generator)
        print(f'{task}, {task.targets[0]}')
    print('==================================================')
