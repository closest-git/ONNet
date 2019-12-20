import torch
import math
import numpy as np
from .some_utils import *

class ChunkPool(torch.nn.Module):
    def __init__(self, nCls,config,pooling="max",chunk_dim=-1):
        super(ChunkPool, self).__init__()
        self.nClass = nCls
        self.pooling = pooling
        self.chunk_dim=chunk_dim
        self.config = config
        #self.regions = split_regions_2d(x.shape,self.nClass)

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
            if self.config.output_chunk=="1D":
                sections=split__sections(x.shape[self.chunk_dim],self.nClass)
                for xx in x.split(sections, self.chunk_dim):
                    x2 = xx.contiguous().view(nSamp, -1)
                    if self.pooling == "max":
                        x3 = torch.max(x2, 1)
                        x_max.append(x3.values)
                    else:
                        x3 = torch.mean(x2, 1)
                        x_max.append(x3)
            else:   #2D
                regions = split_regions_2d(x.shape,self.nClass)
                for box in regions:
                    x2 = x[...,box[0]:box[1],box[2]:box[3]]
                    x2 = x2.contiguous().view(nSamp, -1)
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