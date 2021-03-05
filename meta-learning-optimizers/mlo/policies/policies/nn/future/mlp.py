import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)

from .utils import instantiate_layer, instantiate_activation, initialize_layer
from ..policy import Policy


class MLP(Policy):

    r'''Class for MLP as Policy Model

        Config example:

        policy:
          id: MLP
          blocks:
            -
              layer:
                type: linear
                args:
                  in_features: 14
                  out_features: 32
              activation:
                type: leakyrelu
                args: 
                  negative_slope: 0.01
              initialization:
                type: xavier_uniform
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
            seed: 123141
            repeat_output: 1

         Type options:

             type: linear

             type: bayes_linear

         Activations options:

             type: tanh

             type: softplus
             args:
               beta: 0.01

             type: sigmoid

             type: relu

             type: leakyrelu
             args:
               negative_slope: 0.01

         Initialization options:

             type: uniform
             args:
               a: 0.0
               b: 1.0

             type: normal
             args:
               mean: 0.0
               std: 1.0

             type: constant
             args:
               val: 0.0

             type: xavier_uniform

             type: xavier_normal

             type: kaiming_uniform

             type: kaiming_normal

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

        self.model = nn.Sequential(*torch_layers)

    def forward(self, observation):

        output_size = self.blocks[-1]['layer']['args']['out_features'] * self.repeat_output

        if observation is None:
            population_size = (output_size // self.dim) + 1
            X = np.ones((population_size * self.dim, self.dim)) * (-1.0)
            Y = np.ones(population_size) * (-1.0)
            observation = (X, Y)

        with torch.no_grad():

            # Input:
            X, Y = observation

            # Fitness Shaping (only use sort values):
            idx = np.argsort(Y)
            input_ = X[idx]

            # Output:
            input_ = torch.from_numpy(input_).flatten()

            output = []
            for r in range(self.repeat_output):

                # Sampler:
                prediction = self.model(input_).data.numpy()
                output.append(prediction)

            output = np.array(output)
            output = output.reshape((output_size // self.dim, self.dim))

            return 5.0 * output

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
            p.requires_grad = False

    def reset(self, seed=None):
        return self
