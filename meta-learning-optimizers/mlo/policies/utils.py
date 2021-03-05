def load_policy(config):

    from .policies.random_search import BatchRandomSearch
    from .policies.cmaes import CMAES
    from .policies.nn.lstm_two import LSTMTwo

    policy_classes = {"BatchRandomSearch": BatchRandomSearch,
                      "CMAES": CMAES,
                      "LSTMTwo": LSTMTwo}

    policy = policy_classes[config['id']](config)
    return policy
