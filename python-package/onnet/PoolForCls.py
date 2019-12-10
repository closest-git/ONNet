import torch
from .Z_utils import COMPLEX_utils as Z

#Very strange behavior of DROPOUT
class PoolForCls(torch.nn.Module):
    def __init__(self, nCls,pooling="max"):
        super(PoolForCls, self).__init__()
        self.nClass = nCls
        self.pooling = pooling

    def forward(self, x):
        nSamp = x.shape[0]
        if False:
            x1 = torch.zeros((nSamp, self.nClass)).double()
            step = self.M // self.nClass
            for samp in range(nSamp):
                for i in range(self.nClass):
                    x1[samp,i] = torch.max(x[samp,:,:,i*step:(i+1)*step])
            x_np = x1.detach().cpu().numpy()
            x = x1.cuda()
        else:
            x_max=[]
            for xx in x.chunk(self.nClass, -1):
                x2 = xx.contiguous().view(nSamp, -1)
                if self.pooling == "max":
                    x3 = torch.max(x2, 1)
                    x_max.append(x3.values)
                else:
                    x3 = torch.mean(x2, 1)
                    x_max.append(x3)
            x = torch.stack(x_max,1)
            #x_np = x.detach().cpu().numpy()
            #print(x_np)
        return x