import torch
import math
import numpy as np

class ChunkPool(torch.nn.Module):
    def __init__(self, nCls,pooling="max",chunk_dim=-1):
        super(ChunkPool, self).__init__()
        self.nClass = nCls
        self.pooling = pooling
        self.chunk_dim=chunk_dim

    def __repr__(self):
        main_str = super(ChunkPool, self).__repr__()
        main_str += f"_cls[{self.nClass}]_pool[{self.pooling}]"
        return main_str

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
            split_dim = range(x.shape[self.chunk_dim])
            sections=[]
            for arr in np.array_split(np.array(split_dim), self.nClass):
                sections.append(len(arr))
            #assert split_size>0
            for xx in x.split(sections, self.chunk_dim):
                x2 = xx.contiguous().view(nSamp, -1)
                if self.pooling == "max":
                    x3 = torch.max(x2, 1)
                    x_max.append(x3.values)
                else:
                    x3 = torch.mean(x2, 1)
                    x_max.append(x3)
            assert len(x_max)==self.nClass
            x = torch.stack(x_max,1)
            #x_np = x.detach().cpu().numpy()
            #print(x_np)
        return x

class BinaryChunk(torch.nn.Module):
    def __init__(self, nCls,isLogit=False,pooling="max",chunk_dim=-1):
        super(BinaryChunk, self).__init__()
        self.nClass = nCls
        self.nChunk = (int)(math.ceil(math.log2(self.nClass)))
        self.pooling = pooling
        self.isLogit = isLogit

    def __repr__(self):
        main_str = super(BinaryChunk, self).__repr__()
        if self.isLogit:
            main_str += "_logit"
        main_str += f"_nChunk{self.nChunk}_cls[{self.nClass}]_pool[{self.pooling}]"
        return main_str

    def chunk_poll(self,ck,nSamp):
        x2 = ck.contiguous().view(nSamp, -1)
        if self.pooling == "max":
            x3 = torch.max(x2, 1)
            return x3.values
        else:
            x3 = torch.mean(x2, 1)
            return x3

    def forward(self, x):
        nSamp = x.shape[0]
        x_max=[]
        for ck in x.chunk(self.nChunk, -1):
            if self.isLogit:
                x_max.append(self.chunk_poll(ck,nSamp))
            else:
                for xx in ck.chunk(2, -2):
                    x2 = xx.contiguous().view(nSamp, -1)
                    if self.pooling == "max":
                        x3 = torch.max(x2, 1)
                        x_max.append(x3.values)
                    else:
                        x3 = torch.mean(x2, 1)
                        x_max.append(x3)
        x = torch.stack(x_max,1)

        return x