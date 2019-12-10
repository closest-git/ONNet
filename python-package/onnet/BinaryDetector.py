import torch
from .Z_utils import COMPLEX_utils as Z

#Very strange behavior of DROPOUT
class BinaryDetector(torch.nn.Module):
    def __init__(self, M_in, N_in):
        super(BinaryDetector, self).__init__()
        assert (M_in == N_in)
        self.M = M_in
        self.N = N_in

    def forward(self, x):
        x=torch.mean(x)
        return x