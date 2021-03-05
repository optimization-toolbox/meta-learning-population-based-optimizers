
from .learning import LearningTrainBuilder, LearningValidationBuilder
from .preprocessing import PreprocessingDatageneratorBuilder
from .postprocessing import PostprocessingMetalossesBuilder, PostprocessingECDFSBuilder


class ObjectFactory:

    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)

class RunnerProvider(ObjectFactory):
    def get(self, runner_id, **kwargs):
        return self.create(runner_id, **kwargs)

runner_provider = RunnerProvider()
runner_provider.register_builder('LearningTrain', LearningTrainBuilder())
runner_provider.register_builder('LearningValidation', LearningValidationBuilder())
runner_provider.register_builder('PreprocessingDatagenerator', PreprocessingDatageneratorBuilder())
runner_provider.register_builder('PostprocessingMetalosses', PostprocessingMetalossesBuilder())
runner_provider.register_builder('PostprocessingECDFS', PostprocessingECDFSBuilder())
