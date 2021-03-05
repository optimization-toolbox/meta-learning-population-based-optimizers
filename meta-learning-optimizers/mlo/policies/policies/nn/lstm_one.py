from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)

from .utils import instantiate_layer, instantiate_activation, initialize_layer
from ..policy import Policy

#torch.set_printoptions(precision=10, sci_mode=False)

class LSTMOne(Policy):

    r'''Class for SimpleLSTM as Policy Model

        Config example:

        policy:
          id: LSTMDebugger
          blocks:
          - type: lstm
            args:
              input_size: 3
              hidden_size: 30
              num_layers: 2
            initialization:
              type: normal
              args:
                mean: 0.0
                std: 0.5
          - type: linear
            args:
              in_features: 30
              out_features: 1
            activation:
              type: tanh
            initialization:
              type: normal
              args:
                mean: 0.0
                std: 0.5
          info:
            dim: 2
            seed: 646
            population_size: 6
            repeat_output: 1
    '''

    def __init__(self, config):
        super().__init__(config)

        # Create info:
        self.dim = self.info['dim']
        self.population_size = self.info['population_size']
        self.seed = self.info['seed']

        self.repeat_output = self.info['repeat_output']

        # Create model:
        torch.manual_seed(self.seed)

        torch_layers = []
        for block_config in self.blocks:

            # Instantiate:
            layer = instantiate_layer(block_config)

            # Add Activation:
            activation = instantiate_activation(block_config)

            # Initialize:
            initialize_layer(layer, block_config)

            # Append:
            torch_layers.append(layer)
            if activation is not None:
                torch_layers.append(activation)

        self.lstm = torch_layers[0]
        self.mlp = nn.Sequential(*torch_layers[1:])

        self.model = nn.ModuleDict(OrderedDict({
            'lstm': self.lstm, 'mlp': self.mlp}))

    def forward(self, observation, action, reward, time):

        # Prepare input:
        if (observation is None):
            _input = torch.zeros(self.population_size, self.dim)
        else:
            idxs = np.argsort(observation)
            sorted_action = action[idxs]
            _input = torch.from_numpy(sorted_action)
        _input = _input.flatten()

        with torch.no_grad():

            # Prepare for LSTM:
            _input = _input.view(-1, 1, _input.shape[0])

            # Model Prediction:
            _output, (self._hidden, self._cell) = self.model['lstm'](_input, (self._hidden, self._cell))
            _output = _output[-1, -1]

            final_output = np.empty((self.population_size, self.dim))
            for i in range(self.population_size):
                final_output[i] = self.model['mlp'](_output)
            return final_output

            #_output = self.model['mlp'](_output)
            #_output = np.array(_output).reshape(self.population_size, self.dim)
            #return _output

    def get_params(self):
        parameters = np.concatenate([p.detach().numpy().ravel() for p in self.model.parameters()])
        return parameters

    def set_params(self, new_weights):
        last_slice = 0
        for n, p in self.model.named_parameters():
            size_layer_parameters = np.prod(np.array(p.data.size()))
            new_parameters = new_weights[last_slice:last_slice + size_layer_parameters].reshape(p.data.shape)
            last_slice += size_layer_parameters
            p.data = torch.from_numpy(new_parameters).detach()

    def reset(self, seed=None):

        lstm_num_layers = self.blocks[0]['args']['num_layers']
        _hidden_size = self.blocks[0]['args']['hidden_size']

        # Reset hidden and cell states:
        self._hidden = torch.zeros(lstm_num_layers, 1, _hidden_size)
        self._cell = torch.zeros(lstm_num_layers, 1, _hidden_size)
        return self
