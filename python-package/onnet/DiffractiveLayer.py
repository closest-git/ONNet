import torch
from .Z_utils import COMPLEX_utils as Z
from .some_utils import *
import numpy as np
import random
import torch.nn as nn
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


#https://pytorch.org/tutorials/beginner/pytorch_with_examples.html#pytorch-custom-nn-modules
class DiffractiveLayer(torch.nn.Module):
    def SomeInit(self, M_in, N_in,HZ=0.4e12):
        assert (M_in == N_in)
        self.M = M_in
        self.N = N_in
        self.z_modulus = Z.modulus
        self.size = M_in
        self.delta = 0.03
        self.dL = 0.02
        self.c = 3e8
        self.Hz = HZ#0.4e12

        self.H_z = self.Init_H()

    def __repr__(self):
        #main_str = super(DiffractiveLayer, self).__repr__()
        main_str = f"DiffractiveLayer_[{(int)(self.Hz/1.0e9)}G]_[{self.M},{self.N}]"
        return main_str

    def __init__(self, M_in, N_in,config,HZ=0.4e12):
        super(DiffractiveLayer, self).__init__()
        self.SomeInit(M_in, N_in,HZ)
        assert config is not None
        self.config = config
        #self.init_value = init_value
        #self.rDrop = rDrop
        if not hasattr(self.config,'wavelet') or self.config.wavelet is None:
            if self.config.modulation=="phase":
                self.transmission = torch.nn.Parameter(data=torch.Tensor(self.size, self.size), requires_grad=True)
            else:
                self.transmission = torch.nn.Parameter(data=torch.Tensor(self.size, self.size, 2), requires_grad=True)

            init_param = self.transmission.data
            if self.config.init_value=="reverse":    #
                half=self.transmission.data.shape[-2]//2
                init_param[..., :half, :] = 0
                init_param[..., half:, :] = np.pi
            elif self.config.init_value=="random":
               init_param.uniform_(0, np.pi*2)
            elif self.config.init_value == "random_reverse":
               init_param = torch.randint_like(init_param,0,2)*np.pi
            elif self.config.init_value == "chunk":
                sections = split__sections()
                for xx in init_param.split(sections, -1):
                    xx = random.random(0,np.pi*2)

        #self.rDrop = config.rDrop

        #self.bias = torch.nn.Parameter(data=torch.Tensor(1, 1), requires_grad=True)

    def visualize(self,visual,suffix, params):
        param = self.transmission.data
        name = f"{suffix}_{self.config.modulation}_"
        return visual.image(name,param, params)

    def share_weight(self,layer_1):
        tp = type(self)
        assert(type(layer_1)==tp)
        #del self.transmission
        #self.transmission = layer_1.transmission

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
        u1 = Z.Hadamard(z0,self.H_z.float())
        u2 = Z.fft(u1,"C2C",inverse=True)
        return  u2 * N * N * df * df

    def GetTransCoefficient(self):
        '''
            eps = 1e-5; momentum = 0.1; affine = True

            mean = torch.mean(self.transmission, 1)
            vari = torch.var(self.transmission, 1)
            amp_bn = torch.batch_norm(self.transmission,mean,vari)
        :return:
        '''
        amp_s = Z.exp_euler(self.transmission)

        return amp_s

    def forward(self, x):
        diffrac = self.Diffractive_(x)
        amp_s = self.GetTransCoefficient()
        x = Z.Hadamard(diffrac,amp_s.float())
        if(self.config.rDrop>0):
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

class DiffractiveWavelet(DiffractiveLayer):
    def __init__(self,  M_in, N_in,config,HZ=0.4e12):
        super(DiffractiveWavelet, self).__init__(M_in, N_in,config,HZ)
        #self.hough = torch.nn.Parameter(data=torch.Tensor(2), requires_grad=True)
        self.Init_DisTrans()
        #self.GetXita()

    def __repr__(self):
        main_str = f"Diffrac_Wavelet_[{(int)(self.Hz/1.0e9)}G]_[{self.M},{self.N}]"
        return main_str

    def share_weight(self,layer_1):
        tp = type(self)
        assert(type(layer_1)==tp)
        del self.wavelet
        self.wavelet = layer_1.wavelet
        del self.dis_map
        self.dis_map = layer_1.dis_map
        del self.wav_indices
        self.wav_indices = layer_1.wav_indices


    def Init_DisTrans(self):
        origin_r, origin_c = (self.M-1) / 2, (self.N-1) / 2
        origin_r = random.uniform(0, self.M-1)
        origin_c = random.uniform(0, self.N - 1)
        self.dis_map={}
        #self.dis_trans = torch.zeros((self.size, self.size)).int()
        self.wav_indices = torch.LongTensor((self.size*self.size)).cuda()
        nz=0
        for r in range(self.M):
            for c in range(self.N):
                off = np.sqrt((r - origin_r) * (r - origin_r) + (c - origin_c) * (c - origin_c))
                i_off = (int)(off+0.5)
                if i_off not in self.dis_map:
                    self.dis_map[i_off]=len(self.dis_map)
                id = self.dis_map[i_off]
                #self.dis_trans[r, c] = id
                self.wav_indices[nz] = id;        nz=nz+1
                #print(f"[{r},{c}]={self.dis_trans[r, c]}")
        nD = len(self.dis_map)
        if False:
            plt.imshow(self.dis_trans.numpy())
            plt.show()

        self.wavelet = torch.nn.Parameter(data=torch.Tensor(nD), requires_grad=True)
        self.wavelet.data.uniform_(0, np.pi*2)
        #self.dis_trans = self.dis_trans.cuda()

    def GetXita(self):
        if False:
            xita = torch.zeros((self.size, self.size))
            for r in range(self.M):
                for c in range(self.N):
                    pos = self.dis_trans[r, c]
                    xita[r,c] = self.wavelet[pos]
            origin_r,origin_c=self.M/2,self.N/2
            #xita = self.dis_trans*self.hough[0]+self.hough[1]
        else:
            xita = torch.index_select(self.wavelet, 0, self.wav_indices)
            xita = xita.view(self.size, self.size)

        # print(xita)
        return xita

    def GetTransCoefficient(self):
        xita = self.GetXita()
        amp_s = Z.exp_euler(xita)
        return amp_s

    def visualize(self,visual,suffix, params):
        xita = self.GetXita()
        name = f"{suffix}"
        return visual.image(name,torch.sin(xita.detach()), params)