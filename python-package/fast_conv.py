'''
@Author: Yingshi Chen

@Date: 2020-03-04 14:50:24
@
# Description: 
'''

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
sys.path.append('..')
#from deap.convolve import convDEAP_GIP
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from deap.helpers import getOutputShape
from deap.mappers import PhotonicConvolverMapper
from deap.mappers import ModulatorArrayMapper
from deap.mappers import PWBArrayMapper

class MRMTransferFunction:
    """
    Computes the transfer function of a microring modulator (MRM).
    """
    def __init__(self, a=0.9, r=0.9):
        self.a = a
        self.r = r
        self._maxThroughput = self.throughput(np.pi)

    def throughput(self, phi):
        I_pass = self.a**2 - 2 * self.r * self.a * np.cos(phi) + self.r**2
        I_input = 1 - 2 * self.a * self.r * np.cos(phi) + (self.r * self.a)**2
        return I_pass / I_input

    def phaseFromThroughput(self, Tn):
        Tn = np.asarray(Tn)

        # Create variable to store results
        ans = np.empty_like(Tn)

        # For high throuputs, set to pi
        moreThanMax = Tn >= self._maxThroughput 
        maxOrLess = ~moreThanMax
        ans[moreThanMax] = np.pi

        # Now solve the remainng
        cos_phi = Tn[maxOrLess] * (1 + (self.r * self.a)**2) - self.a**2 - self.r**2  # noqa
        ans[maxOrLess] = np.arccos(cos_phi / (-2 * self.r * self.a * (1 - Tn[maxOrLess])))  # noqa
        #ans = np.arccos(cos_phi / (-2 * self.r * self.a * (1 - Tn[maxOrLess])))

        return ans

def convDEAP(image, kernel, stride, bias=0, normval=255):
    """
    Image is a 3D matrix with index values row, col, depth, index
    Kernel is a 4D matrix with index values row, col, depth, index.
        The depth of the kernel must be equal to the depth of the input.
    """
    assert image.shape[2] == kernel.shape[2]

    # Allocate memory for storing result of convolution
    outputShape = getOutputShape(image.shape, kernel.shape, stride=stride)
    output = np.zeros(outputShape)

    # Build the photonic circuit
    weightBanks = []
    inputShape = (kernel.shape[0], kernel.shape[1])
    for k in range(image.shape[2]):
        pc = PhotonicConvolverMapper.build(
                imageShape=inputShape,
                kernelShape=inputShape,
                power=normval)
        weightBanks.append(pc)

    for k in range(kernel.shape[3]):
        # Load weights
        weights = kernel[:, :, :, k]
        for c in range(weights.shape[2]):
            PWBArrayMapper.updateKernel(
                weightBanks[c].pwbArray,
                weights[:, :, c])

        for h in range(0, outputShape[0], stride):
            for w in range(0, outputShape[1], stride):
                # Load inputs
                inputs = \
                    image[h:min(h + kernel.shape[0], image.shape[0]),
                          w:min(w + kernel.shape[0], image.shape[1]), :]
                for c in range(kernel.shape[2]):
                    ModulatorArrayMapper.updateInputs(
                        weightBanks[c].modulatorArray,
                        inputs[:, :, c],
                        normval=normval)

                # Perform convolution:
                for c in range(kernel.shape[2]):
                    output[h, w, k] += weightBanks[c].step()
                output[h, w, k] += bias

    return output

def convDEAP_GIP(image, kernel, stride, convolverShape=None):
    """
    Image is a 3D matrix with index values row, col, depth, index
    Kernel is a 4D matrix with index values row, col, depth, index.
        The depth of the kernel must be equal to the depth of the input.
    """
    assert image.shape[2] == kernel.shape[2]
    assert kernel.shape[2] == 1 and kernel.shape[3] == 1
    if convolverShape is None:
        convolverShape = image.shape

    # Define convolutional parameters
    Hm, Wm = convolverShape[0], convolverShape[1]
    H, W = image.shape[0], image.shape[1]
    R = kernel.shape[0]

    # Allocate memory for storing result of convolution
    outputShape = getOutputShape(image.shape, kernel.shape, stride=stride)
    output = np.zeros(outputShape)

    # Load weights
    pc = PhotonicConvolverMapper.build(imageShape=convolverShape,kernel=kernel[:, :, 0, 0], power=255)

    input_buffer = np.zeros(convolverShape)
    normval=255
    _mrm = MRMTransferFunction()
    for h in range(0, H - R + 1, Hm - R + 1):
        for w in range(0, W - R + 1, Wm - R + 1):
            inputs = image[h:min(h + Hm, H), w:min(w + Wm, W), 0]
            # Load inputs into a buffer if convolution shape doesn't tile
            # nicely.
            input_buffer[:inputs.shape[0], :inputs.shape[1]] = inputs
            input_buffer[inputs.shape[0]:, inputs.shape[1]:] = 0

            if False:
                ModulatorArrayMapper.updateInputs(pc.modulatorArray,input_buffer,normval=255)
            else:
                #phaseShifts = ModulatorArrayMapper.computePhaseShifts(input_buffer, normval=255)
                normalized = input_buffer / normval
                assert not np.any(input_buffer < 0)                
                phaseShifts = _mrm.phaseFromThroughput(normalized)
                pc.modulatorArray._update(phaseShifts)

            # Perform the convolution and store to memory
            result = pc.step()[:min(h + Hm, H) - h - R + 1,
                               :min(w + Wm, W) - w - R + 1]
            output[h:min(h + Hm, H) - R + 1,
                   w:min(w + Hm, W) - R + 1,
                   0] = result

    return output

def main():
    image = plt.imread("./data/bass.jpg")
    greyscale = np.mean(image, axis=2)

    # Define kernel
    gaussian_kernel = np.zeros((3, 3, 1, 1))
    gaussian_kernel[:, :, 0, 0] = \
            np.array([
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]]) * 1/16


    # Perform convolution
    paddedInputs = np.pad(greyscale, (2, 2), 'constant')
    paddedInputs = np.expand_dims(paddedInputs, 2)
    convolved = convDEAP_GIP(paddedInputs, gaussian_kernel, 1, (12, 12))
    t0=time.time()
    for i in range(10):
        convDEAP_GIP(paddedInputs, gaussian_kernel, 1, (12, 12))
    print(f"convDEAP_GIP T_10={time.time()-t0:.3f}")


    t0=time.time()
    for i in range(10):
        convolve2d(greyscale, gaussian_kernel[:, :, 0, 0])
    print(f"convolve2d T_10={time.time()-t0:.3f}")
    conv_scipy = convolve2d(greyscale, gaussian_kernel[:, :, 0, 0])

    err = np.abs(convolved[:, :, 0] - conv_scipy)
    mse = np.sum(err**2) / (err.size)
    print("MSE distance per pixel", mse)

if __name__ == '__main__':
    main()