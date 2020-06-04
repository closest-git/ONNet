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
from .SparseSupport import *
from .FFT_layer import *
import numpy as np
from .DiffractiveLayer import *
import cv2
useAttention=False
if useAttention:
    import entmax
#from torchscope import scope

class DNET_config:
    def __init__(self,batch,lr_base,modulation="phase",init_value = "random",random_seed=42,
                 support=SuppLayer.SUPP.exp,isFC=False):
        '''

        :param modulation:
        :param init_value: ["random","zero","random_reverse","reverse","chunk"]
        :param support:
        '''
        self.custom_legend = "Express Wavenet"  #"Express_OFF"  "Express Wavenet","Pan_OFF Express_OFF"    #for paper and debug
        self.seed = random_seed
        seed_everything(self.seed)
        self.init_value = init_value  # "random"  "zero"
        self.rDrop = 0
        self.support = support  #None
        self.modulation = modulation    #["phase","phase_amp"]
        self.output_chunk = "2D"        #["1D","2D"]
        self.output_pooling = "max"
        self.batch = batch
        self.learning_rate = lr_base
        self.isFC = isFC
        self.input_scale = 1
        self.wavelet = None              #dict paramter for wavelet
        #if self.isFC == True:            self.learning_rate = lr_base/10
        self.input_plane = ""       #"fourier"

    def env_title(self):
        title=f"{self.support.value}"
        if self.isFC:       title += "[FC]"
        if self.custom_legend is not None:
            title = title + f"_{self.custom_legend}"
        return title

    def __repr__(self):
        main_str = f"lr={self.learning_rate}_ mod={self.modulation} input={self.input_scale} detector={self.output_chunk} " \
            f"support={self.support}"
        if self.isFC:       main_str+=" [FC]"
        if self.custom_legend is not None:
            main_str = main_str + f"_{self.custom_legend}"
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
        if self.config.support == "binary":
            nGate = output.shape[1] // 2
            #assert nGate == self.n
            pred = 0
            for i in range(nGate):
                no = 2*(nGate - 1 - i)
                val_2 = torch.stack([output[:, no], output[:, no + 1]], 1)
                pred_i = val_2.max(1, keepdim=True)[1]  # get the index of the max log-probability
                pred = pred * 2 + pred_i
        elif self.config.support == "logit":
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

    def GetLayer_(self):
        # layer = DiffractiveAMP
        if self.config.wavelet is None:
            layer = DiffractiveLayer
        else:
            layer = DiffractiveWavelet
        return layer

    def __init__(self,IMG_size,nCls,nDifrac,config):
        super(D2NNet, self).__init__()
        self.M,self.N=IMG_size
        self.z_modulus = Z.modulus
        self.nDifrac = nDifrac
        #self.isFC = False
        self.nClass = nCls
        #self.init_value = "random"    #"random"  "zero"
        self.config = config
        self.title = f"DNNet"
        self.highWay = 1        #1,2,3
        if self.config.input_plane == "fourier":            
            self.highWay = 0   

        if hasattr(self.config,'feat_extractor'):
            if self.config.feat_extractor!="last_layer":
                self.feat_extractor = []

        if self.config.output_chunk == "2D":
            assert(self.M*self.N>=self.nClass)
        else:
            assert (self.M >= self.nClass and self.N >= self.nClass)
        print(f"D2NNet nClass={nCls} shape={self.M,self.N}")


        layer = self.GetLayer_()
        #fl = FFT_Layer(self.M, self.N,config,isInv=False)
        self.DD = nn.ModuleList([
            layer(self.M, self.N,config) for i in range(self.nDifrac)
        ])
        if self.config.input_plane=="fourier":
            self.DD.insert(0,FFT_Layer(self.M, self.N,config,isInv=False))
            self.DD.append(FFT_Layer(self.M, self.N,config,isInv=True))
        self.nD = len(self.DD)
        self.laySupp = None

        if self.highWay>0:
            self.wLayer = torch.nn.Parameter(torch.ones(len(self.DD)))
            if self.highWay==2:
                self.wLayer.data.uniform_(-1, 1)
            elif self.highWay==1:
                self.wLayer = torch.nn.Parameter(torch.ones(len(self.DD)))

        #self.DD.append(DropOutLayer(self.M, self.N,drop=0.9999))
        if self.config.isFC:
            self.fc1 = nn.Linear(self.M*self.N, self.nClass)
            self.loss = UserLoss.cys_loss
            self.title = f"DNNet_FC"
        elif self.config.support!=None:
            self.laySupp = SuppLayer(config,self.nClass)
            self.last_chunk = ChunkPool(self.laySupp.nChunk, config, pooling=config.output_pooling)
            self.loss = UserLoss.cys_loss
            a = self.config.support.value
            self.title = f"DNNet_{self.config.support.value}"
        else:
            self.last_chunk = ChunkPool(self.nClass,config,pooling=config.output_pooling)
            self.loss = UserLoss.cys_loss

        if self.config.wavelet is not None:
            self.title = self.title+f"_W"
        if self.highWay>0:
            self.title = self.title + f"_H"
        if self.config.custom_legend is not None:
            self.title = self.title + f"_{self.config.custom_legend}"

        ''' 
        BinaryChunk is pool
        elif self.config.support=="binary":
            self.last_chunk = BinaryChunk(self.nClass, pooling="max")
            self.loss = D2NNet.binary_loss
            self.title = f"DNNet_binary"
        elif self.config.support == "logit":
            self.last_chunk = BinaryChunk(self.nClass, isLogit=True, pooling="max")
            self.loss = D2NNet.logit_loss
        '''

    def visualize(self,visual,suffix):
        no = 0
        for plot in visual.plots:
            images,path = [],""
            if plot['object']=='layer pattern':
                path = f"{visual.img_dir}/{suffix}.jpg"
                for no,layer in enumerate(self.DD):
                    info = f"{suffix},{no}]"
                    title = f"layer_{no+1}"
                    if self.highWay==2:
                        a = self.wLayer[no]
                        a = torch.sigmoid(a)
                        info = info+f"_{a:.2g}"
                    elif self.highWay==1:
                        a = self.wLayer[no]
                        info = info+f"_{a:.2g}"
                        title  = title+f" w={a:.2g}"
                    image = layer.visualize(visual,info,{'save':False,'title':title})
                    images.append(image)
                    no=no+1
            if len(images)>0:
                image_all = np.concatenate(images, axis=1)
            #cv2.imshow("", image_all);    cv2.waitKey(0)
                cv2.imwrite(path,image_all)

    def legend(self):
        if self.config.custom_legend is not None:
            leg_ = self.config.custom_legend
        else:
            leg_ = self.title
        return leg_

    def __repr__(self):
        main_str = super(D2NNet, self).__repr__()
        main_str += f"\n========init={self.config.init_value}"
        return main_str

    def input_trans(self,x):    # square-rooted and normalized
        #x = x.double()*self.config.input_scale
        if True:
            x = x*self.config.input_scale
            x_0,x_1 = torch.min(x).item(),torch.max(x).item()
            assert x_0>=0
            x = torch.sqrt(x)
        else:       #为何不行，莫名其妙
            x = Z.exp_euler(x*2*math.pi).float()
            x_0,x_1 = torch.min(x).item(),torch.max(x).item()
        return x

    def do_classify(self,x):
        if self.config.isFC:
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            return x

        x = self.last_chunk(x)
        if self.laySupp != None:
            x = self.laySupp(x)
        # output = F.log_softmax(x, dim=1)
        return x

    def OnLayerFeats(self):
        pass

    def forward(self, x):
        if hasattr(self, 'feat_extractor'):
            self.feat_extractor.clear()
        nSamp,nChannel = x.shape[0],x.shape[1]
        assert(nChannel==1)
        if nChannel>1:
            no = random.randint(0,nChannel-1)
            x = x[:,0:1,...]
        x = self.input_trans(x)
        if hasattr(self,'visual'):            self.visual.onX(x.cpu(), f"X@input")
        summary = 0
        for no,layD in enumerate(self.DD):
            info = layD.__repr__()
            x = layD(x)
            if hasattr(self,'feat_extractor'):
                self.feat_extractor.append((self.z_modulus(x),self.wLayer[no]))
            if hasattr(self,'visual'):         self.visual.onX(x,f"X@{no+1}")
            if self.highWay==2:
                s = torch.sigmoid(self.wLayer[no])
                summary+=x*s
                x = x*(1-s)
            elif self.highWay==1:
                summary += x * self.wLayer[no]
            elif self.highWay==3:
                summary += self.z_modulus(x) * self.wLayer[no]
        if self.highWay==2:
            x=x+summary
            x = self.z_modulus(x)
        elif self.highWay == 1:
            x = summary
            x = self.z_modulus(x)
        elif self.highWay == 3:
            x = summary
        elif self.highWay == 0:
            x = self.z_modulus(x)
        if hasattr(self,'visual'):            self.visual.onX(x,f"X@output")


        if hasattr(self,'feat_extractor'):
            return
        elif hasattr(self.config,'feat_extractor') and self.config.feat_extractor=="last_layer":
            return x
        else:
            output = self.do_classify(x)
            return output

class MultiDNet(D2NNet):
    def __init__(self, IMG_size,nCls,nInterDifrac,freq_list,config,shareWeight=True):
        super(MultiDNet, self).__init__(IMG_size,nCls,nInterDifrac,config)
        self.isShareWeight=shareWeight
        self.freq_list = freq_list
        nFreq = len(self.freq_list)
        del self.DD;     self.DD = None
        self.wFreq = torch.nn.Parameter(torch.ones(nFreq))
        layer = self.GetLayer_()
        self.freq_nets=nn.ModuleList([
            nn.ModuleList([
                layer(self.M, self.N, self.config, HZ=freq) for i in range(self.nDifrac)
            ]) for freq in freq_list
        ])
        if self.isShareWeight:
            nSubNet = len(self.freq_nets)
            net_0 = self.freq_nets[0]
            for i in range(1,nSubNet):
                net_1 = self.freq_nets[i]
                for j in range(self.nDifrac):
                    net_1[j].share_weight(net_0[j])


    def legend(self):
        if self.config.custom_legend is not None:
            leg_ = self.config.custom_legend
        else:
            title = f"MF_DNet({len(self.freq_list)} channels)"
        return title

    def __repr__(self):
        main_str = super(MultiDNet, self).__repr__()
        main_str += f"\nfreq_list={self.freq_list}_"
        return main_str

    def forward(self, x0):
        nSamp = x0.shape[0]
        x_sum = 0
        for id,fNet in enumerate(self.freq_nets):
            x = self.input_trans(x0)
            #d0,d1=x0.min(),x0.max()
            #x = x0.double()
            for layD in fNet:
                x = layD(x)
            #x_sum = torch.max(x_sum,self.z_modulus(x))).values()
            x_sum += self.z_modulus(x)*self.wFreq[id]
        x = x_sum

        output = self.do_classify(x)
        return output

def main():
    pass

if __name__ == '__main__':
    main()