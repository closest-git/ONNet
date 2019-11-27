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

class BaseNet(nn.Module):
    def __init__(self):
        super(D2NNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
class DiffractiveLayer(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DiffractiveLayer, self).__init__()
        self.size = 512
        self.delta = 0.03
        self.dL = 0.02
        self.c = 3e8
        self.Hz = 0.4e12

    def Diffractive_(self,u0,  theta=0.0):
        d=self.delta, N=self.size, dL=self.dL, lmb=self.c / self.Hz
        # Parameter
        df = 1.0 / dL
        k = np.pi * 2.0 / lmb
        D = dL * dL / (N * lmb)

        # phase
        def phase(i, j):
            i -= N // 2
            j -= N // 2
            return ((i * df) * (i * df) + (j * df) * (j * df))

        ph = np.fromfunction(phase, shape=(N, N), dtype=np.float32)
        # H
        H = np.exp(1.0j * k * d) * np.exp(-1.0j * lmb * np.pi * d * ph)
        # Result
        u1 = torch.fft(u0)
        return torch.ifft(np.fft.fftshift(H) * u1 * dL * dL / (N * N)) * N * N * df * df

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred

class D2NNet(nn.Module):
    # https://github.com/szagoruyko/diracnets

    def __init__(self):
        super(D2NNet, self).__init__()

        self.fc1 = DiffractiveLayer(512, 512)
        self.fc2 = DiffractiveLayer(512, 512)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def main():
    pass

if __name__ == '__main__':
    main()