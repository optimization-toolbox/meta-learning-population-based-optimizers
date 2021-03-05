from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)

from .utils import instantiate_layer, instantiate_activation, initialize_layer
from ..policy import Policy

class SimpleLSTM(Policy):

    r'''Class for SimpleLSTM as Policy Model

        Config example:

        policy:
          id: SimpleLSTM
          blocks:
            -
              layer:
                type: lstm
                args:
                  input_size: 28
                  hidden_size: 64
                  num_layers: 2
              initialization:
                type: xavier_uniform
            -
              layer:
                type: linear
                args:
                  in_features: 64
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
        self.model = nn.Sequential(*torch_layers[1:])

    def forward(self, observation, reward, time, action):

        output_size = self.blocks[-1]['layer']['args']['out_features'] * self.repeat_output

        if observation is None and action is None:

            # Observation:
            state_size = (output_size) + self.dim
            X = np.ones((state_size + self.dim, self.dim)) * (-1.0)
            Y = np.ones(state_size // self.dim) * (-1.0)
            observation = (X, Y)

            # Action:
            action = np.ones((output_size // self.dim, self.dim)) * (-1.0)

        with torch.no_grad():

            # Input:

            # observation:
            X, Y = observation

            # reward and time:
            reward = torch.from_numpy(np.array([reward], dtype=np.float64))
            time = torch.from_numpy(np.array([time]))

            # action:
            action = torch.from_numpy(action).flatten()

            # Fitness Shaping (only use sort values):
            idx = np.argsort(Y)
            state = X[idx]
            state = torch.from_numpy(state).flatten()

            # Final Input:
            input_lstm = torch.cat([state, reward, time, action])
            input_lstm = input_lstm.view(-1, 1, input_lstm.shape[0])

            # LSTM forward:
            lstm_output, (self.lstm_hidden, self.lstm_cell) = self.lstm(input_lstm, (self.lstm_hidden, self.lstm_cell))
            lstm_output = lstm_output[-1, -1]

            output = []
            for r in range(self.repeat_output):

                # Sampler:
                prediction = self.model(lstm_output).data.numpy()
                output.append(prediction)

            output = np.array(output)
            output = output.reshape((output_size // self.dim, self.dim))

            return 5.0 * output

    def get_params(self):
        parameters = np.concatenate([p.detach().numpy().ravel() for p in self.model.parameters()])
        return parameters

    def set_params(self, new_weights):
        last_slice = 0
        for p in self.model.parameters():
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
