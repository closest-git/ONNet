# Authors: Yingshi Chen(gsp.cys@gmail.com)

'''
    PyTorch implementation of D2CNN     ------      All-optical machine learning using diffractive deep neural networks
'''

import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.nn.functional as F
from .Z_utils import COMPLEX_utils as Z
from .PoolForCls import *
from .Loss import *
import numpy as np
from .DiffractiveLayer import *

class D2NNet(nn.Module):
    # https://github.com/szagoruyko/diracnets

    def __init__(self,IMG_size,nCls,nDifrac):
        super(D2NNet, self).__init__()
        self.M,self.N=IMG_size
        self.z_modulus = Z.modulus
        self.nDifrac = nDifrac
        self.isFC = False
        self.nClass = nCls
        self.loss = UserLoss.cys_loss
        assert(self.M>=self.nClass and self.N>=self.nClass)
        print(f"D2NNet nClass={nCls} shape={self.M,self.N}")

        #layer = DiffractiveAMP
        layer = DiffractiveLayer
        self.DD = nn.ModuleList([
            layer(self.M, self.N) for i in range(self.nDifrac)
        ])
        self.nD = len(self.DD)
        #self.DD.append(DropOutLayer(self.M, self.N,drop=0.9999))
        if self.isFC:
            self.fc1 = nn.Linear(self.M*self.N, self.nClass)
        else:
            self.last_pool = PoolForCls(self.nClass,pooling="max")
        #total = sum([param.nelement() for param in self.parameters()])
        #print(f"nParameters={total}")#\nparams={self.parameters()}
        #print(self)

    def __repr__(self):
        main_str = super(D2NNet, self).__repr__()
        return main_str

    def forward(self, x):
        x = x.double()
        for layD in self.DD:
            x = layD(x)

        x = self.z_modulus(x).cuda()
        if self.isFC:
            x = torch.flatten(x, 1)
            x = self.fc1(x)
        else:
            x = self.last_pool(x)

        output = F.log_softmax(x, dim=1)
        return output

    def predict(self,output):
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        return pred

def main():
    pass

if __name__ == '__main__':
    main()