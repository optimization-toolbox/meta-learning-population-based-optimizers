import torch
import torch.nn as nn
import torch.nn.init as init

def instantiate_layer(block_config):

    # Read Type:
    curr_type = block_config['type']

    # Build layer:
    if curr_type == 'linear':
        layer = nn.Linear(**block_config['args'])
    elif curr_type == 'bayes_linear':
        layer = BayesLinear(**block_config['args'])
    elif curr_type == 'lstm':
        layer = nn.LSTM(**block_config['args'])#, batch_first=True)
    else:
        print("instantiate_layer problem!")
        exit()
    return layer

def instantiate_activation(block_config):

    if block_config['type'] == 'lstm':
        return None

    # Read Type:
    curr_type = block_config['activation']['type']

    # Build activation:
    if curr_type == 'tanh':
        activation = nn.Tanh()
    elif curr_type == 'softplus':
        activation = nn.Softplus(**block_config['activation']['args'])
    elif curr_type == 'sigmoid':
        activation = nn.Sigmoid()
    elif curr_type == 'relu':
        activation = nn.ReLU()
    elif curr_type == 'leakyrelu':
        activation = nn.LeakyReLU(**block_config['activation']['args'])
    elif curr_type == None:
        activation = None
    else:
        print("instantiate_activation problem!")
        exit()
    return activation


def initialize_layer(layer, block_config):

    # Read Type:
    curr_type = block_config['initialization']['type']

    # Initialize layer::
    if curr_type == 'default':
        return

    for n, param in layer.named_parameters():
        if 'bias' in n:
            init.zeros_(param.data)
        if 'weight' or 'bias' in n:
            if curr_type == 'uniform':
                init.uniform_(param.data, **block_config['initialization']['args'])
            elif curr_type == 'normal':
                init.normal_(param.data, **block_config['initialization']['args'])
            elif curr_type == 'constant':
                init.constant_(param.data, **block_config['initialization']['args'])
            elif 'xavier' in curr_type:
                gain = calculate_gain(block_config)
                if curr_type == 'xavier_uniform':
                    init.xavier_uniform_(param.data, gain)
                elif curr_type == 'xavier_normal':
                    init.xavier_normal_(param.data, gain)
            elif 'kaiming' in curr_type:

                if block_config['activation']['type'] == 'leakyrelu':
                    args = {'a': block_config['activation']['args']['negative_slope'], 'nonlinearity': 'leaky_relu'}
                elif block_config['activation']['type'] == 'relu':
                    args = {'nonlinearity': 'relu'}

                if curr_type == 'kaiming_uniform':
                    init.kaiming_uniform_(param.data, **args)
                elif curr_type == 'kaiming_normal':
                    init.kaiming_normal_(param.data, **args)
            else:
                print("initialize_layer problem!")
                exit()

def calculate_gain(block_config):

    if block_config['type'] == 'lstm':
        activation_type = 'tanh'
    else:
        activation_type = block_config['activation']['type']

    if activation_type in ['sigmoid', 'tanh', 'relu']:
        return init.calculate_gain(activation_type)
    elif activation_type == 'leakyrelu':
        return init.calculate_gain('leaky_relu', block_config['activation']['args']['negative_slope'])
    else:
        return 1.0


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
