from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)

from .utils import instantiate_layer, instantiate_activation, initialize_layer
from ..policy import Policy

#torch.set_printoptions(precision=10, sci_mode=False)

class LSTMTwo(Policy):

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
        self.multiply_output = self.info.get('multiply_output')
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

        with torch.no_grad():

            # Prepare input:
            if (observation is None):
                params = np.zeros((self.population_size, self.dim))
                ranks = np.zeros(self.population_size)
            else:

                idxs = np.argsort(observation)
                ranks = np.linspace(0, 1, self.population_size)
                ranks = ranks[idxs]
                params = action

            batch = np.empty((1, self.population_size*self.dim, 2))
            for i in range(self.population_size):
                for j in range(self.dim):
                    param = params[i][j]
                    rank = ranks[i]
                    batch[0][i*self.dim + j] = np.array((param, rank))
            batch = torch.from_numpy(batch)

            _output, (self._hidden, self._cell) = self.model['lstm'](batch, (self._hidden, self._cell))
            _output = _output[0]
            
            final_output = self.model['mlp'](_output).numpy()
            final_output = final_output.reshape((self.population_size, self.dim))

            if self.multiply_output != None:
                return final_output * self.multiply_output 
            else:
                return final_output


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

        gen = torch.Generator()
        gen = gen.manual_seed(624234)

        # Reset hidden and cell states:
        self._hidden = torch.rand(lstm_num_layers, self.population_size*self.dim, _hidden_size, generator=gen)# * rs_.uniform(-1, 1)
        self._cell = torch.rand(lstm_num_layers, self.population_size*self.dim, _hidden_size, generator=gen)# * rs_.uniform(-1, 1)

        #print(self._hidden)
        #exit()
        return self
