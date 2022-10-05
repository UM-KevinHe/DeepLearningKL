import torch
from torch import nn
from torchtuples import tuplefy


class DenseVanillaBlock(nn.Module):
    def __init__(self, in_features, out_features, bias=True, batch_norm=True, dropout=0., activation=nn.ReLU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        if w_init_:
            w_init_(self.linear.weight.data)
        self.activation = activation()
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, input):
        input = self.activation(self.linear(input))
        if self.batch_norm:
            input = self.batch_norm(input)
        if self.dropout:
            input = self.dropout(input)
        return input


class ProportionalBlock(nn.Module):
    """
        Simulate the last hidden layer in the proportional design. The goal for this operation is
        to exploit the node (X^T b) and apply the formulas of 3 link functions.
    """
    def __init__(self, out_features, option):
        """
        :param out_features: The dimension of the output layer (also that of the output layer for the whole
        neural network).
        :param option: The option for the link functions.
        "log-log": The log-log link function.
        "log-log-2" The log-log link function (implementation following from nnet-survival)
        "log": The log link function.
        "logit": The logit link function.
        """
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((1, out_features)))
        nn.init.kaiming_normal_(self.a.data, nonlinearity='relu')
        self.option = option

    def forward(self, input):
        option = self.option
        if option == "log-log":
            input = input + torch.log(-torch.log(torch.sigmoid(self.a)))
            input = 1 - torch.exp(-torch.exp(input))
            return input
        elif option == "log-log-2":
            input = torch.pow(torch.sigmoid(self.a), torch.exp(input))
            input = 1 - input
            return input
        elif option == "log":
            input = input + torch.log(torch.sigmoid(self.a))
            Softplus = nn.Softplus()
            input = Softplus(input)
            return input
        elif option == "logit":
            return torch.sigmoid(input + self.a)


class DenseProportionalBlock(nn.Module):
    """
    Simulate the second last hidden layer in the proportional design. The goal for this operation is
    to combine all the effects of the input layer into one node.
    Illustration: (X (R^n), b (R^n) ) -> X^T b (R)
    """
    def __init__(self, in_features,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, False)
        if w_init_:
            w_init_(self.linear.weight.data)

    def forward(self, input):
        input = self.linear(input)
        return input


class MLPVanilla(nn.Module):
    """
    Vanilla MLP, simulating the fully-connected neural network.
    """
    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        """
        :param in_features: The dimension of the input vector.
        :param num_nodes: The number of nodes (dimension) for each hidden layer.
        :param out_features: The dimension of the output vector.
        :param batch_norm: Whether applying batch normalization in each layer.
        :param dropout: Whether applying dropout in each layer.
        :param activation: The activation function in each hidden layer (we recommend ReLU and not change it).
        :param output_activation: The activation function for the output layer.
        :param output_bias: Whether there will be bias term in the output layer.
        :param w_init_: The initialization technique.
        """
        super().__init__()
        num_nodes = tuplefy(in_features, num_nodes).flatten()
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes) - 1)]
        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))
        net.append(nn.Linear(num_nodes[-1], out_features, output_bias))
        if output_activation:
            net.append(output_activation)
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)


class MLPProportional(nn.Module):
    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu'), option=None):
        """
        :param in_features: The dimension of the input vector.
        :param num_nodes: The number of nodes (dimension) for each hidden layer.
        :param out_features: The dimension of the output vector.
        :param batch_norm: Whether applying batch normalization in each layer.
        :param dropout: Whether applying dropout in each layer.
        :param activation: The activation function in each hidden layer (we recommend ReLU and not change it).
        :param output_activation: The activation function for the output layer.
        :param output_bias: Whether there will be bias term in the output layer.
        :param w_init_: The initialization technique.
        :param option: Whether the link function is.
        This will take effect in the neural network structure "ProportionalBlock".
        """
        if option not in ['log-log', 'log', 'logit', 'log-log-2']:
            raise ValueError("Please provide with a link function, it should be one of ['log-log', 'log-log-2', 'log', 'logit']")
        super().__init__()
        num_nodes = tuplefy(in_features, num_nodes).flatten()
        if not hasattr(dropout, '__iter__'):
            dropout = [dropout for _ in range(len(num_nodes) - 1)]
        net = []
        for n_in, n_out, p in zip(num_nodes[:-1], num_nodes[1:], dropout):
            net.append(DenseVanillaBlock(n_in, n_out, True, batch_norm, p, activation, w_init_))

        # New Change!
        net.append(DenseProportionalBlock(num_nodes[-1]))
        net.append(ProportionalBlock(out_features, option))
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)