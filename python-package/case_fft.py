from torchvision import datasets, transforms
from PIL import Image
import  numpy as np
import matplotlib.pyplot as plt
from onnet import *
import torch
from skimage import io, transform
torch.set_printoptions(profile="full")

size = 28
delta = 0.03
dL = 0.02
c = 3e8
Hz = 0.4e12

def Init_H(d=delta, N = size, dL = dL, lmb = c/Hz,theta=0.0):
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
    H_f = np.fft.fftshift(H)
    #print(H_f);    print(H)
    return H,H_f

def fft_test(H_f,N = 28):
    dL = 0.02
    s = dL * dL / (N * N)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        #transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #normalize
    ])
    image = io.imread("E:/ONNet/data/MNIST/test_2.jpg").astype(np.float64)
    #print(image)
    img_tensor = torch.from_numpy(image)
    #print(img_tensor)
    #img_tensor.unsqueeze_(0)
    print(img_tensor.shape, img_tensor.dtype)
    u0 = COMPLEX_utils.ToZ(img_tensor)
    print(u0.shape, H_f.shape);

    u1 = COMPLEX_utils.fft(u0)
    print(u1)
    H_z = np.zeros(H_f.shape + (2,))
    H_z[..., 0] = H_f.real
    H_z[..., 1] = H_f.imag
    H_f = torch.from_numpy(H_z)
    u1 = COMPLEX_utils.Hadamard(H_f,u1)     #H_f * u1
    print(u1)
    u1 = COMPLEX_utils.fft(u1 ,"C2C",inverse=True)
    print(u1)
    input(...)

if __name__ == '__main__':
    H, H_f = Init_H()
    fft_test(H_f)