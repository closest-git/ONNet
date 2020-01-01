import torch
from .D2NNet import *
from .some_utils import *
import numpy as np
import random
import torch.nn as nn
from enum import Enum

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
class SuppLayer(torch.nn.Module):
    class SUPP(Enum):
        exp,sparse,expW,diff = 'exp','sparse','expW','differentia'

    def __init__(self,config,nClass, nSupp=10):
        super(SuppLayer, self).__init__()
        self.nClass = nClass
        self.nSupp = nSupp
        self.nChunk = self.nClass*2
        self.config = config
        self.w_11=False
        if self.config.support==self.SUPP.sparse:   #"supp_sparse":
            if self.w_11:
                tSupp = torch.ones(self.nClass, self.nSupp)
            else:
                tSupp = torch.Tensor(self.nClass, self.nSupp).uniform_(-1,1)
            self.wSupp = torch.nn.Parameter(tSupp)
            self.nChunk = self.nSupp*self.nSupp
            self.chunk_map = np.random.randint(self.nChunk, size=(self.nClass, self.nSupp))
        #elif self.config.support=="supp_expW":
        #    self.nSupp = 2
        #    self.wSupp = torch.nn.Parameter(torch.ones(2))

    def __repr__(self):
        w_init="1" if self.w_11 else "random"
        main_str = f"SupportLayer supp=({self.nSupp},{w_init}) type=\"{self.config.support}\" nChunk={self.nChunk}"
        return main_str

    def sparse_support(self,x):
        feats=[]
        for i in range(self.nClass):
            feat = 0;
            for j in range(self.nSupp):
                col = (int)(self.chunk_map[i,j])
                feat += x[:, col]*self.wSupp[i,j]
            feats.append(torch.exp(feat))      #why exp is useful???
            #feats.append(feat)
        output = torch.stack(feats,1)
        return output

    def forward(self, x):
        if self.config.support == self.SUPP.sparse:     # "supp_sparse":
            output = self.sparse_support(x)
            return output

        assert x.shape[1] == self.nClass * 2
        if self.config.support==self.SUPP.diff:     #"supp_differentia":
            for i in range(self.nClass):
                x[:,i] = (x[:,2*i]-x[:,2*i+1])/(x[:,2*i]+x[:,2*i+1])
            output=x[...,0:self.nClass]
        elif self.config.support==self.SUPP.exp:     #"supp_exp":
            for i in range(self.nClass):
                x[:, i] = torch.exp(x[:, 2 * i] - x[:, 2 * i + 1])
            output = x[..., 0:self.nClass]
        elif self.config.support==self.SUPP.expW:     #"supp_expW":
            output = torch.zeros_like(x)
            for i in range(self.nClass):
                output[:, i] = torch.exp(x[:, 2 * i]*self.w2[0] - x[:, 2 * i + 1]*self.w2[1])
            output = output[..., 0:self.nClass]

        return output

