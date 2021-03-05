from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
torch.set_default_dtype(torch.float64)

from ..policy import Policy


class StackedLSTM(Policy):
    r"""Class for Deep Neural Network as Policy Model"""

    def __init__(self, config):
        print("stacked_lstm not implemented!")
        exit()
        super().__init__(config)

        # 'global_seed': 4

        # Encoder:
        # 'enc_dim': 2
        # 'enc_layers': [14, 32, 12]
        # 'enc_noise': [False, False, False]
        # 'enc_activations': ['tanh', 'tanh', None]
        # 'enc_initialization_method': 'xavier'
        # 'enc_initialization_param': None
        # 'enc_repeat_output' : 1
        # 'enc_global_seed': 42

        # LSTM_1:
        # lstm_1_hidden_size: 64
        # lstm_1_num_layers: 1

        # LSTM_2:
        # lstm_2_hidden_size: 64
        # lstm_2_num_layers: 1

        # Decoder:
        # 'dec_dim': 2
        # 'dec_layers': [14, 32, 12]
        # 'dec_noise': [False, False, False]
        # 'dec_activations': ['tanh', 'tanh', None]
        # 'dec_initialization_method': 'xavier'
        # 'dec_initialization_param': None
        # 'dec_repeat_output' : 1
        # 'dec_global_seed': 42

        # Create encoder:
        enc_torch_layers = []
        for i in range(1, len(self.enc_layers)):

            # Add Layer:
            if self.enc_noise[i - 1] == False:
                layer = nn.Linear(self.enc_layers[i - 1], self.enc_layers[i])
            else:
                layer = BayesLinear(self.enc_layers[i - 1], self.enc_layers[i])
            enc_torch_layers.append(layer)

            # Add Activation:
            if self.enc_activations[i - 1] == None:
                continue
            elif self.enc_activations[i - 1] == 'tanh':
                activation = nn.Tanh()
            elif self.enc_activations[i - 1] == 'sigmoid':
                activation = nn.Sigmoid()
            enc_torch_layers.append(activation)
        self.enc = nn.Sequential(*enc_torch_layers)

        # Create LSTM_1:
        self.lstm_1 = nn.LSTM(self.enc_layers[-1] + 2, self.lstm_1_hidden_size, self.lstm_1_num_layers, batch_first=True)
        # +2 because of reward and time

        # Create decoder:
        dec_torch_layers = []
        for i in range(1, len(self.dec_layers)):

            # Add Layer:
            if self.enc_noise[i - 1] == False:
                layer = nn.Linear(self.dec_layers[i - 1], self.dec_layers[i])
            else:
                layer = BayesLinear(self.dec_layers[i - 1], self.dec_layers[i])
            dec_torch_layers.append(layer)

            # Add Activation:
            if self.enc_activations[i - 1] == None:
                continue
            elif self.enc_activations[i - 1] == 'tanh':
                activation = nn.Tanh()
            elif self.enc_activations[i - 1] == 'sigmoid':
                activation = nn.Sigmoid()
            dec_torch_layers.append(activation)
        self.dec = nn.Sequential(*dec_torch_layers)

        # Create LSTM_2:
        _in_size = self.enc_layers[-1] + self.lstm_1_hidden_size + (self.dec_layers[-1] * self.dec_repeat_output)
        self.lstm_2 = nn.LSTM(_in_size, self.lstm_2_hidden_size, self.lstm_2_num_layers, batch_first=True)

        # Model:
        self.model = nn.ModuleDict(OrderedDict({
            'enc': self.enc,
            'lstm_1': self.lstm_1,
            'lstm_2': self.lstm_2,
            'dec': self.dec
        }))

        # Init weights:
        # torch.manual_seed(self.global_seed)
        # self.encoder.apply(self.weight_init)

    def weight_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, BayesLinear):
            for n, param in m.named_parameters():
                if 'bias' in n:
                    init.zeros_(param.data)
                elif 'weight' in n:
                    if self.initialization_method == 'uniform':
                        init.uniform_(param.data,
                                      -self.initialization_param,
                                      self.initialization_param)
                    elif self.initialization_method == 'normal':
                        init.normal_(param.data, mean=0.0, std=self.initialization_param)
                    elif self.initialization_method == 'xavier':
                        gain = init.calculate_gain('sigmoid')
                        init.xavier_normal_(param.data, gain)

    def forward(self, observation, reward, time, action):

        if observation is None and action is None:

            # Observation:
            X = np.ones((self.enc_layers[0] // self.enc_dim, self.enc_dim)) * (-1.0)
            Y = np.ones(self.enc_layers[0] // self.enc_dim) * (-1.0)
            observation = (X, Y)

            # Action:
            action = np.ones(((self.dec_layers[-1] * self.dec_repeat_output) // self.dec_dim, self.dec_dim)) * (-1.0)

        # print("!")
        # print(action.shape)
        # print(observation[0].shape)
        # print(observation[1].shape)
        # exit()

        with torch.no_grad():

            # Input:

            # observation
            X, Y = observation

            # reward and time:
            reward = torch.from_numpy(np.array([reward], dtype=np.float64))  # .double()
            time = torch.from_numpy(np.array([time]))  # .double()

            # action
            action = torch.from_numpy(action).flatten()
            #prev_action = torch.from_numpy(prev_action).float().flatten()

            # Encode:

            # Fitness Shaping (only use sort values):
            idx = np.argsort(Y)
            input_ = X[idx]
            input_ = torch.from_numpy(input_).flatten()
            encoded = self.model['enc'](input_)  # .data.numpy()

            # LSTM_1:
            encoded_reward_time = torch.cat([encoded, reward, time])
            encoded_reward_time = encoded_reward_time.view(-1, 1, encoded_reward_time.shape[0])

            lstm_1_output, (self.lstm_1_hidden, self.lstm_1_cell) = self.model['lstm_1'](encoded_reward_time, (self.lstm_1_hidden, self.lstm_1_cell))
            lstm_1_output = lstm_1_output[-1, -1]

            # LSTM_2:
            encoded_outlstm1_action = torch.cat([encoded, lstm_1_output, action])
            encoded_outlstm1_action = encoded_outlstm1_action.view(-1, 1, encoded_outlstm1_action.shape[0])

            lstm_2_output, (self.lstm_2_hidden, self.lstm_2_cell) = self.model['lstm_2'](encoded_outlstm1_action, (self.lstm_2_hidden, self.lstm_2_cell))
            lstm_2_output = lstm_2_output[-1, -1]

            # Sampler:
            prediction = torch.Tensor(self.dec_repeat_output, self.dec_dim)
            for i in range(len(prediction)):
                prediction[i] = self.model['dec'](lstm_2_output)

            return prediction.numpy() * 5.0

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

        # Reset hidden and cell states of offspring_generator
        self.lstm_1_hidden = torch.zeros(self.lstm_1_num_layers, 1, self.lstm_1_hidden_size)
        self.lstm_1_cell = torch.zeros(self.lstm_1_num_layers, 1, self.lstm_1_hidden_size)
        self.lstm_2_hidden = torch.zeros(self.lstm_2_num_layers, 1, self.lstm_2_hidden_size)
        self.lstm_2_cell = torch.zeros(self.lstm_2_num_layers, 1, self.lstm_2_hidden_size)
        return self


class BayesLinear(nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias = bias
        if bias:
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))
        else:
            print("BayesLinear: Bias should be True")
            exit()

    def forward(self, input):

        weight = self.weight_mu

        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_log_sigma)

        return nn.functional.linear(input, weight, bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(self.in_features, self.out_features, self.bias is not None)
