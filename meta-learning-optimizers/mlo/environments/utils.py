def load_environment(config):

    from .learning_to_optimize_POMDP import LearningToOptimizePOMDP

    environment_classes = {'LearningToOptimizePOMDP': LearningToOptimizePOMDP}

    environment = environment_classes[config['id']](config)

    return environment