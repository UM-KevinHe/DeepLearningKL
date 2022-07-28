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


class ProportionalBlock(torch.nn.Module):
    def __init__(self, out_features, option):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn((1, out_features)))
        nn.init.kaiming_normal_(self.a.data, nonlinearity='relu')
        self.option = option

    def forward(self, input):
        option = self.option
        if (option == "log-log"):
            input = input + torch.log(-torch.log(torch.sigmoid(self.a)))
            input = 1 - torch.exp(-torch.exp(input))
            return input
        elif (option == "log"):
            input = input + torch.log(torch.sigmoid(self.a))
            input = torch.exp(input)
            return input
        elif (option == "logit"):
            return torch.sigmoid(input + self.a)


class DenseProportionalBlock(nn.Module):
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
    def __init__(self, in_features, num_nodes, out_features, batch_norm=True, dropout=None, activation=nn.ReLU,
                 output_activation=None, output_bias=True,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
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
        if option not in ['log-log', 'log', 'logit']:
            raise ValueError("Please provide with a link function, it should be one of ['log-log', 'log', 'logit']")
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
        # New
        net.append(DenseProportionalBlock(num_nodes[-1]))
        net.append(ProportionalBlock(out_features, option))
        self.net = nn.Sequential(*net)

    def forward(self, input):
        return self.net(input)