# Authors: Yingshi Chen(gsp.cys@gmail.com)

'''
    PyTorch implementation of D2CNN     ------      All-optical machine learning using diffractive deep neural networks
'''

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import  numpy as np
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
class DiffractiveLayer(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(DiffractiveLayer, self).__init__()
        self.size = 512
        self.delta = 0.03
        self.dL = 0.02
        self.c = 3e8
        self.Hz = 0.4e12
        self.amp = torch.nn.Parameter(data=torch.Tensor(self.size, self.size), requires_grad=True)
        self.amp.data.uniform_(0, 1)

    def Diffractive_(self,u0,  theta=0.0):
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
        # Result
        u1 = torch.fft(u0,2)
        return torch.ifft(np.fft.fftshift(H) * u1 * self.dL * self.dL / (N * N)) * N * N * df * df

    def forward(self, x):
        x = self.Diffractive_(x)*self.amp
        return x

class D2NNet(nn.Module):
    # https://github.com/szagoruyko/diracnets

    def __init__(self):
        super(D2NNet, self).__init__()

        self.fc1 = DiffractiveLayer(512, 512)
        self.fc2 = DiffractiveLayer(512, 512)
        print(self.parameters())
        print(self)


    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def main():
    pass

if __name__ == '__main__':
    main()