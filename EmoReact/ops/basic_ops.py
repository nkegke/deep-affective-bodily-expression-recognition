import torch
import math
import warnings
warnings.filterwarnings("ignore")

class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class SegmentConsensus(torch.autograd.Function):
    @staticmethod
    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        output = input_tensor.mean(dim=1, keepdim=True)
        return output
    
    @staticmethod
    def backward(self, grad_output):
        grad_in = grad_output.expand(self.shape) / float(self.shape[1])
        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus(self.consensus_type, self.dim).apply(input)
