from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)

from .utils import instantiate_layer, instantiate_activation, initialize_layer
from ..policy import Policy

torch.set_printoptions(precision=10, sci_mode=False)

class LSTMDebugger(Policy):

    r'''Class for SimpleLSTM as Policy Model

        Config example:

        policy:
          id: MyLSTM
          blocks:
            -
              layer:
                type: lstm
                args:
                  input_size: (6) + (6)*2 + 1 + 1 = 20
                  hidden_size: 32
                  num_layers: 2
              initialization:
                type: default
          info:
            dim: 2
            population_size: 6
            seed: 123141
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

        #print(observation)

        if (observation is None):

            # Observation:
            observation = np.linspace(0, -1, self.population_size)

            # Action:
            action = np.array([np.linspace(0, -1, self.dim).tolist()] * self.population_size)

            # Reward:
            reward = 0.0

            #print(observation)
            #print(action)
            #print(reward)

        # exit()

        # if (observation is None) and (action is None) and (reward is None) and (time is None):

        #         # Observation:
        #     observation = np.zeros(self.population_size)

        #     # Action:
        #     action = np.zeros((self.population_size, self.dim))

        #     # Reward and Time:
        #     reward, time = 0.0, 0.0

        with torch.no_grad():

            reward = torch.from_numpy(np.array([reward], dtype=np.float64))
            action = torch.from_numpy(action)
            observation = torch.from_numpy(observation)

            population_output = np.zeros((self.population_size, self.dim))

            for l in range(self.population_size):

                row = []

                for d in range(self.dim):
        
                    # Prepare input:
                    _input = torch.tensor([observation[l], action[l,d] , reward])
                    _input = _input.view(-1, 1, _input.shape[0])

                    # Model prediction:
                    _output, (self._hidden[l,d], self._cell[l,d]) = self.model['lstm'](_input, (self._hidden[l,d], self._cell[l,d]))
                    _output = _output[-1, -1]
                    _output = self.model['mlp'](_output)
                    _output = np.array(_output)
                    population_output[l,d] = _output

            return population_output
            print(population_output)

            print(population_output)
            exit()

            # Input:
            observation = np.argsort(observation) / (len(observation) - 1)
            observation = torch.from_numpy(observation).flatten()

            # reward and time:
            reward = torch.from_numpy(np.array([reward], dtype=np.float64))
            time = torch.from_numpy(np.array([time], dtype=np.float64))

            # action:
            action = torch.from_numpy(action).flatten()

            # LSTM:
            input_lstm = torch.cat([observation, action, reward, time])
            input_lstm = input_lstm.view(-1, 1, input_lstm.shape[0])

            # LSTM forward:
            print('POLICY:')
            print(f'LSTM INPUT: {input_lstm}')
            _output, (self._hidden, self._cell) = self.model['lstm'](input_lstm, (self._hidden, self._cell))
            print(f'LSTM OUTPUT: {_output}')
            exit()
            _output = _output[-1, -1]

            _output = np.array(_output)
            _output = _output.reshape((self.population_size, self.dim))

            # print('\n\n')
            # print(f'input_lstm: {input_lstm}')
            # print(f'observation: {observation}')
            # print(f'action: {action}')
            # print(f'reward: {reward}')
            # print(f'time: {time}')
            # print(f'_output: {_output}')
            # print(f'output: {output}')

            return _output

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
        self._hidden = torch.zeros(self.population_size, self.dim, lstm_num_layers, 1, _hidden_size)
        self._cell = torch.zeros(self.population_size, self.dim, lstm_num_layers, 1, _hidden_size)
        return self
