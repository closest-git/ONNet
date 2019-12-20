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

class DNET_config:
    def __init__(self,batch,modulation="phase",init_value = "random",chunk=""):
        '''

        :param modulation:
        :param init_value: ["random","zero","random_reverse","reverse","chunk"]
        :param chunk:
        '''
        self.init_value = init_value  # "random"  "zero"
        self.rDrop = 0
        self.chunk = chunk
        self.modulation = modulation
        self.output_chunk = "2D"        #["1D","2D"]
        self.output_pooling = "max"
        self.batch = batch
        self.learning_rate = 0.01
        self.isFC = False
        self.input_scale = 255
        if self.isFC == True:
            self.learning_rate = 0.0001

    def __repr__(self):
        main_str = f"lr={self.learning_rate}_ mod={self.modulation} input={self.input_scale} detector={self.output_chunk}"
        if self.isFC:       main_str+=" [FC]"
        return main_str


class D2NNet(nn.Module):
    @staticmethod
    def binary_loss(output, target, reduction='mean'):
        nSamp = target.shape[0]
        nGate = output.shape[1] // 2
        loss = 0
        for i in range(nGate):
            target_i = target % 2
            val_2 = torch.stack([output[:,2*i],output[:,2*i+1]],1)

            loss_i = F.cross_entropy(val_2, target_i, reduction=reduction)
            loss += loss_i
            target = (target - target_i) / 2

        # loss = F.nll_loss(output, target, reduction=reduction)
        return loss

    @staticmethod
    def logit_loss(output, target, reduction='mean'):   #https://stackoverflow.com/questions/53628622/loss-function-its-inputs-for-binary-classification-pytorch
        nSamp = target.shape[0]
        nGate = output.shape[1]
        loss = 0
        loss_BCE = nn.BCEWithLogitsLoss()
        for i in range(nGate):
            target_i = target % 2
            out_i = output[:,i]
            loss_i = loss_BCE(out_i, target_i.double())
            loss += loss_i
            target = (target - target_i) / 2
        return loss

    def predict(self,output):
        if self.config.chunk == "binary":
            nGate = output.shape[1] // 2
            #assert nGate == self.n
            pred = 0
            for i in range(nGate):
                no = 2*(nGate - 1 - i)
                val_2 = torch.stack([output[:, no], output[:, no + 1]], 1)
                pred_i = val_2.max(1, keepdim=True)[1]  # get the index of the max log-probability
                pred = pred * 2 + pred_i
        elif self.config.chunk == "logit":
            nGate = output.shape[1]
            # assert nGate == self.n
            pred = 0
            for i in range(nGate):
                no = nGate - 1 - i
                val_2 = F.sigmoid(output[:, no])
                pred_i = (val_2+0.5).long()
                pred = pred * 2 + pred_i
        else:
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        return pred

    def __init__(self,IMG_size,nCls,nDifrac,config):
        super(D2NNet, self).__init__()
        self.M,self.N=IMG_size
        self.z_modulus = Z.modulus
        self.nDifrac = nDifrac
        #self.isFC = False
        self.nClass = nCls
        #self.init_value = "random"    #"random"  "zero"
        self.config = config

        #self.chunk = chunk
        assert(self.M>=self.nClass and self.N>=self.nClass)
        print(f"D2NNet nClass={nCls} shape={self.M,self.N}")

        #layer = DiffractiveAMP
        layer = DiffractiveLayer
        self.DD = nn.ModuleList([
            layer(self.M, self.N,config) for i in range(self.nDifrac)
        ])
        self.nD = len(self.DD)
        #self.DD.append(DropOutLayer(self.M, self.N,drop=0.9999))
        if self.config.isFC:
            self.fc1 = nn.Linear(self.M*self.N, self.nClass)
            self.loss = UserLoss.cys_loss
        elif self.config.chunk=="binary":
            self.last_chunk = BinaryChunk(self.nClass, pooling="max")
            self.loss = D2NNet.binary_loss
        elif self.config.chunk == "logit":
            self.last_chunk = BinaryChunk(self.nClass,isLogit=True, pooling="max")
            self.loss = D2NNet.logit_loss
        else:
            self.last_chunk = ChunkPool(self.nClass,config,pooling=config.output_pooling)
            self.loss = UserLoss.cys_loss

        #total = sum([param.nelement() for param in self.parameters()])
        #print(f"nParameters={total}")#\nparams={self.parameters()}
        #print(self)

    def __repr__(self):
        main_str = super(D2NNet, self).__repr__()
        main_str += f"\n========init={self.config.init_value}"
        return main_str

    def input_trans(self,x):    # square-rooted and normalized
        x = x.double()*self.config.input_scale
        x = torch.sqrt(x)
        return x

    def do_classify(self,x):
        if self.config.isFC:
            x = torch.flatten(x, 1)
            x = self.fc1(x)
        else:
            x = self.last_chunk(x)

        if self.config.chunk=="binary":
            output = x
        else:
            output = x
            # output = F.log_softmax(x, dim=1)
        return output

    def forward(self, x):
        x = self.input_trans(x)

        for layD in self.DD:
            x = layD(x)

        x = self.z_modulus(x).cuda()
        output = self.do_classify(x)
        if False:
            if self.config.isFC:
                x = torch.flatten(x, 1)
                x = self.fc1(x)
            else:
                x = self.last_chunk(x)

            if self.config.chunk=="binary":
                output = x
            else:
                output = x
                #output = F.log_softmax(x, dim=1)
        return output



def main():
    pass

if __name__ == '__main__':
    main()