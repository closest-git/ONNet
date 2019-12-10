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
import  numpy as np

import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
class DiffractiveLayer(torch.nn.Module):
    def __init__(self, M_in, N_in,rDrop=0.0):
        super(DiffractiveLayer, self).__init__()
        assert(M_in==N_in)
        self.M=M_in
        self.N=N_in
        self.z_modulus = Z.modulus
        self.size = M_in
        self.delta = 0.03
        self.dL = 0.02
        self.c = 3e8
        self.Hz = 0.4e12
        self.amp = torch.nn.Parameter(data=torch.Tensor(self.size, self.size,2), requires_grad=True)
        self.amp.data.uniform_(0, 1)
        self.rDrop = rDrop

    def Diffractive_(self,u0,  theta=0.0):
        if Z.isComplex(u0):
            z0 = u0
        else:
            z0 = u0.new_zeros(u0.shape + (2,))
            z0[...,0] = u0

        d=self.delta
        N=self.size
        lmb=self.c / self.Hz
        # Parameter
        df = 1.0 / self.dL
        k = np.pi * 2.0 / lmb
        D = self.dL * self.dL / (N * lmb)

        # phase
        def phase(i, j):
            i -= N // 2
            j -= N // 2
            return ((i * df) * (i * df) + (j * df) * (j * df))

        ph = np.fromfunction(phase, shape=(N, N), dtype=np.float32)
        # H
        H = np.exp(1.0j * k * d) * np.exp(-1.0j * lmb * np.pi * d * ph)
        Hshift = np.fft.fftshift(H)*self.dL*self.dL/(N*N)
        H_z = np.zeros(Hshift.shape + (2,))
        H_z[..., 0] = Hshift.real
        H_z[..., 1] = Hshift.imag
        H_z = torch.from_numpy(H_z).cuda()
        z0 = Z.fft(z0)
        u1 = Z.Hadamard(z0,H_z)
        u2 = Z.fft(u1,"C2C",inverse=True)
        return  u2 * N * N * df * df

    def forward(self, x):
        #x = self.Diffractive_(x)*self.amp
        diffrac = self.Diffractive_(x)
        #self.amp[3] = 0
        x = Z.Hadamard(diffrac,self.amp)
        if(self.rDrop>0):
            drop = Z.rDrop2D(1-self.rDrop,(self.M,self.N),isComlex=True)
            x = Z.Hadamard(x, drop)
        return x

class D2NNet(nn.Module):
    # https://github.com/szagoruyko/diracnets

    def __init__(self,nCls):
        super(D2NNet, self).__init__()
        self.M=28;      self.N=28
        self.z_modulus = Z.modulus
        self.isFC = False
        self.nClass = nCls
        assert(self.M>=self.nClass and self.N>=self.nClass)

        layer = nn.Linear
        layer = DiffractiveLayer
        self.DD = nn.ModuleList([
            layer(self.M, self.N),
            layer(self.M, self.N),
            layer(self.M, self.N),
            layer(self.M, self.N),
            layer(self.M, self.N)]
        )
        self.nD = len(self.DD)
        #self.DD.append(DropOutLayer(self.M, self.N,drop=0.9999))
        if self.isFC:
            self.fc1 = nn.Linear(self.M*self.N, self.nClass)
        else:
            self.last_pool = PoolForCls(self.nClass,pooling="max")
        print(self.parameters())
        print(self)

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

def main():
    pass

if __name__ == '__main__':
    main()