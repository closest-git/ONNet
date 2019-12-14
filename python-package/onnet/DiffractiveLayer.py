import torch
from .Z_utils import COMPLEX_utils as Z
from .some_utils import *
import numpy as np
import random


#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
class DiffractiveLayer(torch.nn.Module):
    def SomeInit(self, M_in, N_in):
        assert (M_in == N_in)
        self.M = M_in
        self.N = N_in
        self.z_modulus = Z.modulus
        self.size = M_in
        self.delta = 0.03
        self.dL = 0.02
        self.c = 3e8
        self.Hz = 0.4e12

        self.H_z = self.Init_H()


    def __init__(self, M_in, N_in,init_value="zero",rDrop=0.0,params="phase"):
        super(DiffractiveLayer, self).__init__()
        self.SomeInit(M_in, N_in)
        self.init_value = init_value
        self.rDrop = rDrop
        if params=="phase":
            self.transmission = torch.nn.Parameter(data=torch.Tensor(self.size, self.size), requires_grad=True)
        else:
            self.transmission = torch.nn.Parameter(data=torch.Tensor(self.size, self.size, 2), requires_grad=True)

        init_param = self.transmission.data
        if self.init_value=="reverse":    #
            half=self.transmission.data.shape[-2]//2
            init_param[..., :half, :] = 0
            init_param[..., half:, :] = np.pi
        elif self.init_value=="random":
           self.transmission.data.uniform_(0, np.pi*2)
        elif self.init_value == "random_reverse":
           init_param = torch.randint_like(init_param,0,2)*np.pi
        elif self.init_value == "chunk":
            sections = split__sections()
            for xx in init_param.split(sections, -1):
                xx = random.random(0,np.pi*2)

        self.rDrop = rDrop

        #self.bias = torch.nn.Parameter(data=torch.Tensor(1, 1), requires_grad=True)

    def Init_H(self):
        # Parameter
        N = self.size
        df = 1.0 / self.dL
        d=self.delta
        lmb=self.c / self.Hz
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
        H_f = np.fft.fftshift(H)*self.dL*self.dL/(N*N)
        # print(H_f);    print(H)
        H_z = np.zeros(H_f.shape + (2,))
        H_z[..., 0] = H_f.real
        H_z[..., 1] = H_f.imag
        H_z = torch.from_numpy(H_z).cuda()
        return H_z

    def Diffractive_(self,u0,  theta=0.0):
        if Z.isComplex(u0):
            z0 = u0
        else:
            z0 = u0.new_zeros(u0.shape + (2,))
            z0[...,0] = u0

        N = self.size
        df = 1.0 / self.dL

        z0 = Z.fft(z0)
        u1 = Z.Hadamard(z0,self.H_z)
        u2 = Z.fft(u1,"C2C",inverse=True)
        return  u2 * N * N * df * df

    def GetTransCoefficient(self):
        amp_s = Z.exp_euler(self.transmission)
        return amp_s

    def forward(self, x):
        diffrac = self.Diffractive_(x)
        amp_s = self.GetTransCoefficient()
        x = Z.Hadamard(diffrac,amp_s)
        if(self.rDrop>0):
            drop = Z.rDrop2D(1-self.rDrop,(self.M,self.N),isComlex=True)
            x = Z.Hadamard(x, drop)
        #x = x+self.bias
        return x

class DiffractiveAMP(DiffractiveLayer):
    def __init__(self, M_in, N_in,rDrop=0.0):
        super(DiffractiveAMP, self).__init__(M_in, N_in,rDrop,params="amp")
        #self.amp = torch.nn.Parameter(data=torch.Tensor(self.size, self.size, 2), requires_grad=True)
        self.transmission.data.uniform_(0, 1)

    def GetTransCoefficient(self):
        # amp_s = Z.sigmoid(self.amp)
        # amp_s = torch.clamp(self.amp, 1.0e-6, 1)
        amp_s = self.transmission
        return amp_s