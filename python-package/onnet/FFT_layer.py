'''
@Author: Yingshi Chen

@Date: 2020-04-10 11:22:27
@
# Description: 
'''

import torch
from .Z_utils import COMPLEX_utils as Z
from .some_utils import *
import numpy as np
import random
import torch.nn as nn
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class FFT_Layer(torch.nn.Module):
    def SomeInit(self, M_in, N_in,isInv=False):
        assert (M_in == N_in)
        self.M = M_in
        self.N = N_in
        self.isInv = isInv

    def __repr__(self):
        i_ = "_i" if self.isInv else ""
        main_str = f"FFT_Layer{i_}_[{self.M},{self.N}]"
        return main_str

    def __init__(self, M_in, N_in,config,isInv=False):
        super(FFT_Layer, self).__init__()
        self.SomeInit(M_in, N_in,isInv)
        assert config is not None
        self.config = config
        #self.init_value = init_value
        
    def visualize(self,visual,suffix, params):
        param = self.transmission.data
        name = f"{suffix}_{self.config.modulation}_"
        return visual.image(name,param, params)


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

    def forward(self, x):
        #return x
        if Z.isComplex(x):
            z0 = x
        else:
            z0 = x.new_zeros(x.shape + (2,))
            z0[...,0] = x
        if self.isInv:
            x = Z.fft(z0,"C2C",inverse=self.isInv)
        else:
            x = (Z.fft(z0,"C2C",inverse=self.isInv))
        x_0,x_1 = torch.min(x),torch.max(x)
        return x
    
    def trans(img):
        plt.figure(figsize=(10,8))
        plt.subplot(121),plt.imshow(img, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        f = (abs(np.fft.fftshift(fftn(img))))**0.25*(255)**3  # Amplify
        plt.subplot(122),plt.imshow(f, cmap = 'gray')
        plt.title('Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()