import math
from src.utils import config
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_feats = in_features
        self.out_feats = out_features
        if config.cuda:
            self.weight = Parameter(torch.rand(in_features, out_features)).to('cuda')
        else:
            self.weight = Parameter(torch.rand(in_features, out_features))
        if bias:
            if config.cuda:
                self.bias = Parameter(torch.rand(out_features)).to('cuda')
            else:
                self.bias = Parameter(torch.rand(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_feats) + ' -> ' \
               + str(self.out_feats) + ')'
