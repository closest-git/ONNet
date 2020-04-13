'''
    1 晕   Pytorch居然不支持复向量 https://github.com/pytorch/pytorch/issues/755

'''

import torch
from torch.nn import ReflectionPad2d
from torch.nn.functional import relu, max_pool2d, dropout, dropout2d
import numpy as np

class COMPLEX_utils(object):
    @staticmethod
    def isComplex(input):
        return input.size(-1) == 2

    @staticmethod
    def isReal(input):
        return input.size(-1) == 1

    @staticmethod
    def ToZ(u0):
        if COMPLEX_utils.isComplex(u0):
            return u0
        else:
            z0 = u0.new_zeros(u0.shape + (2,))
            z0[..., 0] = u0
            assert(COMPLEX_utils.isComplex(z0))
            return z0

    @staticmethod
    def relu(input_r,input_i):
        return relu(input_r), relu(input_i)

    @staticmethod
    def max_pool2d(input_r,input_i,kernel_size, stride=None, padding=0,
                                    dilation=1, ceil_mode=False, return_indices=False):

        return max_pool2d(input_r, kernel_size, stride, padding, dilation,
                          ceil_mode, return_indices), \
               max_pool2d(input_i, kernel_size, stride, padding, dilation,
                          ceil_mode, return_indices)

    @staticmethod
    def rDrop2D(rDrop,d_shape,isComlex=False):
        drop = np.random.binomial(1, rDrop, size=d_shape).astype(np.float)
        drop[drop == 0] = 1.0e-6
        # print(f"x={x.shape} drop={drop.shape}")
        drop = torch.from_numpy(drop).cuda()
        if isComlex:
            drop = COMPLEX_utils.ToZ(drop)
        return drop
    '''
    @staticmethod
    def dropout(input_r,input_i, p=0.5, training=True, inplace=False):
        return dropout(input_r, p, training, inplace), \
               dropout(input_i, p, training, inplace)

    @staticmethod
    def dropout2d(input_r,input_i, p=0.5, training=True, inplace=False):
        return dropout2d(input_r, p, training, inplace), \
               dropout2d(input_i, p, training, inplace)
    '''

    #the absolute value or modulus of z     https://en.wikipedia.org/wiki/Absolute_value#Complex_numbers
    @staticmethod
    def modulus(x):
        shape = x.size()[:-1]
        if False:
            norm = torch.zeros(shape)
            if x.dtype==torch.float64:
                norm = norm.double()
        norm = (x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1]).sqrt()
        return norm

    @staticmethod
    def phase(x):
        phase = torch.atan2(x[..., 0],x[..., 1])
        return phase

    @staticmethod
    def sigmoid(x):
        # norm[...,0] = (x[...,0]*x[...,0] + x[...,1]*x[...,1]).sqrt()
        s_ = torch.zeros_like(x)
        s_[...,0] = torch.sigmoid(x[...,0])
        s_[..., 1] = torch.sigmoid(x[..., 1])
        return s_

    @staticmethod
    def exp_euler(x):       #Euler's formula:   {\displaystyle e^{ix}=\cos x+i\sin x,}
        s_ = torch.zeros(x.shape + (2,)).double().cuda()
        s_[..., 0] = torch.cos(x)
        s_[..., 1] = torch.sin(x)
        return s_

    @staticmethod
    def fft(input, direction='C2C', inverse=False):
        """
            Interface with torch FFT routines for 2D signals.

            Example
            -------
            x = torch.randn(128, 32, 32, 2)
            x_fft = fft(x, inverse=True)

            Parameters
            ----------
            input : tensor
                complex input for the FFT
            direction : string
                'C2R' for complex to real, 'C2C' for complex to complex
            inverse : bool
                True for computing the inverse FFT.
                NB : if direction is equal to 'C2R', then the transform
                is automatically inverse.
        """
        if direction == 'C2R':
            inverse = True

        if not COMPLEX_utils.isComplex(input):
            raise(TypeError('The input should be complex (e.g. last dimension is 2)'))

        if (not input.is_contiguous()):
            raise (RuntimeError('Tensors must be contiguous!'))

        if direction == 'C2R':
            output = torch.irfft(input, 2, normalized=False, onesided=False)*input.size(-2)*input.size(-3)
        elif direction == 'C2C':
            if inverse:
                #output = torch.ifft(input, 2, normalized=False)*input.size(-2)*input.size(-3)
                output = torch.ifft(input, 2, normalized=False)
            else:
                output = torch.fft(input, 2, normalized=False)

        return output

    @staticmethod
    def Hadamard(A, B, inplace=False):
        """
            Complex pointwise multiplication between (batched) tensor A and tensor B.
            Sincr The Hadamard product is commutative, so Hadamard(A, B)=Hadamard(B, A)

            Parameters
            ----------
            A : tensor
                A is a complex tensor of size (B, C, M, N, 2)
            B : tensor
                B is a complex tensor of size (M, N, 2) or real tensor of (M, N, 1)
            inplace : boolean, optional
                if set to True, all the operations are performed inplace

            Returns
            -------
            C : tensor
                output tensor of size (B, C, M, N, 2) such that:
                C[b, c, m, n, :] = A[b, c, m, n, :] * B[m, n, :]
        """
        if not COMPLEX_utils.isComplex(A):
            raise TypeError('The input must be complex, indicated by a last '
                            'dimension of size 2')

        if B.ndimension() != 3:
            raise RuntimeError('The filter must be a 3-tensor, with a last '
                               'dimension of size 1 or 2 to indicate it is real '
                               'or complex, respectively')

        if not COMPLEX_utils.isComplex(B) and not COMPLEX_utils.isReal(B):
            raise TypeError('The filter must be complex or real, indicated by a '
                            'last dimension of size 2 or 1, respectively')

        if A.size()[-3:-1] != B.size()[-3:-1]:
            raise RuntimeError('The filters are not compatible for multiplication!')

        if A.dtype is not B.dtype:
            raise RuntimeError('A and B must be of the same dtype')

        if A.device.type != B.device.type:
            raise RuntimeError('A and B must be of the same device type')

        if A.device.type == 'cuda':
            if A.device.index != B.device.index:
                raise RuntimeError('A and B must be on the same GPU!')

        if COMPLEX_utils.isReal(B):
            if inplace:
                return A.mul_(B)
            else:
                return A * B
        else:
            C = A.new(A.size())

            A_r = A[..., 0].contiguous().view(-1, A.size(-2)*A.size(-3))
            A_i = A[..., 1].contiguous().view(-1, A.size(-2)*A.size(-3))

            B_r = B[...,0].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_i)
            B_i = B[..., 1].contiguous().view(B.size(-2)*B.size(-3)).unsqueeze(0).expand_as(A_r)

            C[..., 0].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_r - A_i * B_i
            C[..., 1].view(-1, C.size(-2)*C.size(-3))[:] = A_r * B_i + A_i * B_r

            return C if not inplace else A.copy_(C)

def IFFT(X1,X2,X3):
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row',figsize=(10,6))
    Z = ifftn(X1)
    ax1.imshow(X1, cmap=cm.Reds)
    ax4.imshow(np.real(Z), cmap=cm.gray)
    Z = ifftn(X2)
    ax2.imshow(X2, cmap=cm.Reds)
    ax5.imshow(np.real(Z), cmap=cm.gray)
    Z = ifftn(X3)
    ax3.imshow(X3, cmap=cm.Reds)
    ax6.imshow(np.real(Z), cmap=cm.gray)
    plt.show()


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

