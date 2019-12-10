import torch
from .Z_utils import COMPLEX_utils as Z

#Very strange behavior of DROPOUT
class DropOutLayer(torch.nn.Module):
    def __init__(self, M_in, N_in,drop=0.5):
        super(DropOutLayer, self).__init__()
        assert (M_in == N_in)
        self.M = M_in
        self.N = N_in
        self.rDrop = drop

    def forward(self, x):
        assert(Z.isComplex(x))
        nX = x.numel()//2
        d_shape=x.shape[:-1]
        drop = np.random.binomial(1, self.rDrop, size=d_shape).astype(np.float)
        #print(f"x={x.shape} drop={drop.shape}")
        drop = torch.from_numpy(drop).cuda()
        x[...,0] *= drop
        x[...,1] *= drop
        return x