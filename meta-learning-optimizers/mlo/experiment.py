
import yaml

from .runners.provider import runner_provider

class Experiment:

    def __init__(self, config_path):

        with open(f'{config_path}') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def run(self):
        runner = runner_provider.get(self.config['runner']['id'], **self.config)
        runner.run()










