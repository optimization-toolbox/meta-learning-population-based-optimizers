from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)

from .utils import instantiate_layer, instantiate_activation, initialize_layer
from ..policy import Policy

class MyLSTM(Policy):

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
            -
              layer:
                type: linear
                args:
                  in_features: 32
                  out_features: 12
              activation:
                type: tanh
              initialization:
                type: normal
                args:
                  mean: 0.0
                  std: 1.0
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
            'lstm': self.lstm,
            'mlp': self.mlp}))

    def forward(self, observation, action, reward, time):

        if (observation is None) and (action is None) and (reward is None) and (time is None):

                # Observation:
            observation = np.zeros(self.population_size)

            # Action:
            action = np.zeros((self.population_size, self.dim))

            # Reward and Time:
            reward, time = 0.0, 0.0

        with torch.no_grad():

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
            lstm_output, (self.lstm_hidden, self.lstm_cell) = self.model['lstm'](input_lstm, (self.lstm_hidden, self.lstm_cell))
            lstm_output = lstm_output[-1, -1]

            output = []
            for r in range(self.repeat_output):

                # Sampler:
                prediction = self.model['mlp'](lstm_output).data.numpy()
                output.append(prediction)

            output = np.array(output)
            output = output.reshape((self.population_size, self.dim))

            # print('\n\n')
            # print(f'input_lstm: {input_lstm}')
            # print(f'observation: {observation}')
            # print(f'action: {action}')
            # print(f'reward: {reward}')
            # print(f'time: {time}')
            # print(f'lstm_output: {lstm_output}')
            # print(f'output: {output}')

            return output

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

        lstm_num_layers = self.blocks[0]['layer']['args']['num_layers']
        lstm_hidden_size = self.blocks[0]['layer']['args']['hidden_size']

        # Reset hidden and cell states:
        self.lstm_hidden = torch.zeros(lstm_num_layers, 1, lstm_hidden_size)
        self.lstm_cell = torch.zeros(lstm_num_layers, 1, lstm_hidden_size)
        return self
