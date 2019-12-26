import torch
from .D2NNet import *
from .some_utils import *
import numpy as np
import random
import torch.nn as nn

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
class SuppLayer(torch.nn.Module):
    def __init__(self,config,nClass, nSupp=10):
        super(SuppLayer, self).__init__()
        self.nClass = nClass
        self.nSupp = nSupp
        self.config = config
        if self.config.support=="supp_sparse":
            self.wSupp = torch.nn.Parameter(torch.ones(self.nClass, self.nSupp))
        #elif self.config.support=="supp_expW":
        #    self.nSupp = 2
        #    self.wSupp = torch.nn.Parameter(torch.ones(2))

    def __repr__(self):
        main_str = f"SuppLayer_[self.nSupp]"
        return main_str

    def forward(self, x):
        if self.config.support=="supp_differentia":
            assert x.shape[1]==self.nClass*2
            for i in range(self.nClass):
                x[:,i] = (x[:,2*i]-x[:,2*i+1])/(x[:,2*i]+x[:,2*i+1])
            output=x[...,0:self.nClass]
        elif self.config.support=="supp_exp":
            assert x.shape[1]==self.nClass*2
            for i in range(self.nClass):
                x[:, i] = torch.exp(x[:, 2 * i] - x[:, 2 * i + 1])
            output = x[..., 0:self.nClass]
        elif self.config.support=="supp_expW":
            assert x.shape[1]==self.nClass*2
            output = torch.zeros_like(x)
            for i in range(self.nClass):
                output[:, i] = torch.exp(x[:, 2 * i]*self.w2[0] - x[:, 2 * i + 1]*self.w2[1])
            output = output[..., 0:self.nClass]
        return output

