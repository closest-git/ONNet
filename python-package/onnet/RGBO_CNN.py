import sys
import torch.nn as nn
import os
#from torchvision import models
sys.path.append("../..")
from cnn_models import *
from torchvision import transforms
from torchvision.transforms.functional import to_grayscale
from torch.autograd import Variable
# from resnet import resnet50
from copy import deepcopy
import numpy as np
import pickle
from .NET_config import *
from .D2NNet import *

class RGBO_CNN_config(NET_config):
    def __init__(self, net_type, data_set, IMG_size, lr_base, batch_size, nClass, nLayer):
        super(RGBO_CNN_config, self).__init__(net_type, data_set, IMG_size, lr_base, batch_size,nClass,nLayer)
        #self.dnet_type = ""
        self.dnet_type = "stack_input"
        self.dnet_type = "stack_feature"

def image_transformer():
    """
    :return:  A transformer to convert a PIL image to a tensor image
              ready to feed into a neural network
    """
    return {
        'train': transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

'''
    1 参见cifar_rgbF.jpg，简单的fourier channel没啥效果
'''
class D_input(nn.Module):
    def __init__(self, config, DNet):
        super(D_input, self).__init__()
        self.config = config
        self.DNet = DNet
        self.inplanes = 64
        self.nLayD = DNet.nDifrac#self.DNet.config.nLayer
        #self.nLayD = 1
        #self.c_input =nn.Conv2d(3+self.nLayD, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.c_input = nn.Conv2d(3+self.nLayD, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        nChan = x.shape[1]
        assert nChan==3 or nChan==1
        if nChan==3:
            gray = x[:, 0:1]*0.3 + 0.59 * x[:, 1:2] + 0.11 * x[:, 2:3]  # to_grayscale(x)
        else:
            gray = x
        return self.DNet.forward(gray)
        listT = []
        for i in range(nChan):
            listT.append(x[:, i:i+1])
        if self.nLayD>=1:
            self.DNet.forward(gray)
            assert len(self.DNet.feat_extractor) == self.nLayD
            for opti, w in self.DNet.feat_extractor:
                listT.append(opti)          #*w
        elif self.nLayD==0:#
            pass
        else:
            listT.append(gray)

        x = torch.stack(listT,dim=1).squeeze()
        if hasattr(self, 'visual'):         self.visual.onX(x, f"D_input")
        x = self.c_input(x)
        return x

    def forward_000(self, x):
        if False:
            gray = x[:, 0:1]  # to_grayscale(x)
            self.DNet.forward(gray)
            # in_opti = self.DNet.concat_layer_modulus()  # self.get_resnet_convs_out(x)
            for opti, w in self.DNet.feat_extractor:
                opti = torch.stack([opti, opti, opti], 1).squeeze()  # opti.repeat(3, 1)
                out_opti = self.resNet.forward(opti)
                out_sum = out_sum + out_opti * w
        pass

class RGBO_CNN(torch.nn.Module):
    '''
        resnet  https://missinglink.ai/guides/pytorch/pytorch-resnet-building-training-scaling-residual-networks-pytorch/
    '''
    def pick_models(self):
        if False:   #from torch vision or cadene models
            model_names = sorted(name for name in cnn_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(models.__dict__[name]))
            print(model_names)
            # pretrainedmodels   https://data.lip6.fr/cadene/pretrainedmodels/
            model_names = ['alexnet', 'bninception', 'cafferesnet101', 'densenet121', 'densenet161', 'densenet169',
                           'densenet201',
                           'dpn107', 'dpn131', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'fbresnet152',
                           'inceptionresnetv2', 'inceptionv3', 'inceptionv4', 'nasnetalarge', 'nasnetamobile',
                           'pnasnet5large',
                           'polynet',
                           'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'resnext101_32x4d',
                           'resnext101_64x4d',
                           'se_resnet101', 'se_resnet152', 'se_resnet50', 'se_resnext101_32x4d', 'se_resnext50_32x4d',
                           'senet154', 'squeezenet1_0', 'squeezenet1_1',
                           'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'xception']

        # model_name='cafferesnet101'
        # model_name='resnet101'
        # model_name='se_resnet50'
        # model_name='vgg16_bn'
        # model_name='vgg11_bn'
        # model_name='dpn68'      #learning rate=0.0001 效果较好
        self.back_bone = 'resnet18_x'
        # model_name='dpn92'
        # model_name='senet154'
        # model_name='densenet121'
        # model_name='alexnet'
        # model_name='senet154'
        cnn_model = ResNet34()          ;#models.resnet18(pretrained=True)
        return cnn_model

    def __init__(self, config,DNet):
        super(RGBO_CNN, self).__init__()
        seed_everything(42)
        self.config = config
        backbone = self.pick_models()
        if self.config.dnet_type == "stack_feature":
            self.DInput = D_input(config,DNet)
        elif self.config.dnet_type == "stack_input":                  #False and hasattr(self,'DInput'):
            self.CNet = nn.Sequential(*list(backbone.children())[1:])
        else:
            self.CNet = nn.Sequential(*list(backbone.children()))

        #print(f"=> creating model CNet='{self.CNet}'\nDNet={self.DNet}")
        if False:   #外层处理
            if config.gpu_device is not None:
                self.cuda(config.gpu_device)
                print(next(self.parameters()).device)
                self.thickness_criterion = self.thickness_criterion.cuda()
                self.metal_criterion = self.metal_criterion.cuda()
            elif config.distributed:
                self.cuda()
                self = torch.nn.parallel.DistributedDataParallel(self)
            else:
                self = torch.nn.DataParallel(self).cuda()

    def save_acti(self,x,name):
        acti = x.cpu().data.numpy()
        self.activations.append({'name':name,'shape':acti.shape,'activation':acti})

#https://forums.fast.ai/t/pytorch-best-way-to-get-at-intermediate-layers-in-vgg-and-resnet/5707/6


    def forward_0(self, x):
        if hasattr(self, 'DInput'):
            x = self.DInput(x)
        for no,lay in enumerate(self.CNet):
            if isinstance(lay,nn.Linear):       #x = self.avgpool(x),        x = x.reshape(x.size(0), -1)
                x = F.avg_pool2d(x, 4)
                x = x.reshape(x.size(0), -1)
            x = lay(x)
            #print(f"{no}:\t{lay}\nx={x}")
            if isinstance(lay,nn.AdaptiveAvgPool2d):       #x = self.avgpool(x),        x = x.reshape(x.size(0), -1)
                x = x.reshape(x.size(0), -1)
        out_sum = x
        return out_sum

    def forward(self, x):
        out_sum = 0
        if self.config.dnet_type == "stack_feature":
            out_sum= self.DInput(x)
        for no,lay in enumerate(self.CNet):
            if isinstance(lay,nn.Linear):       #x = self.avgpool(x),        x = x.reshape(x.size(0), -1)
                x = F.avg_pool2d(x, 4)
                x = x.reshape(x.size(0), -1)
            x = lay(x)
            #print(f"{no}:\t{lay}\nx={x}")
            if isinstance(lay,nn.AdaptiveAvgPool2d):       #x = self.avgpool(x),        x = x.reshape(x.size(0), -1)
                x = x.reshape(x.size(0), -1)
        out_sum += x
        return out_sum

if __name__ == "__main__":
    config = DNET_config(None)
    a = RGBO_CNN(config,nFilmLayer=10)
    print(f"RGBO_CNN={a}")
    pass

