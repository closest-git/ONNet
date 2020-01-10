import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms
from torchvision.transforms.functional import to_grayscale
from torch.autograd import Variable
# from resnet import resnet50
from copy import deepcopy
import numpy as np
import pickle
from .NET_config import *
from .D2NNet import *

class OptiCNN_config(NET_config):
    def __init__(self, net_type, data_set, IMG_size, lr_base, batch_size, nClass, nLayer):
        super(OptiCNN_config, self).__init__(net_type, data_set, IMG_size, lr_base, batch_size,nClass,nLayer)

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

class OptiCNN(torch.nn.Module):
    def pick_models(self):
        model_names = sorted(name for name in models.__dict__
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

    def __init__(self, config,DNet):
        super(OptiCNN, self).__init__()

        self.config = config

        self.pick_models()
        self.resNet = models.resnet18(pretrained=True)
        self.DNet = DNet
        print(f"=> creating model back_bone='{self.resNet}' DNet={self.DNet}")
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

    def get_resnet_all_out(self, x):
        self.activations=[]
        self.save_acti(x, "input")
        x = self.resNet.conv1(x)  # out = [N, 64, 112, 112]
        self.save_acti(x,"conv1")
        x = self.resNet.bn1(x)
        x = self.resNet.relu(x)
        x = self.resNet.maxpool(x)  # out = [N, 64, 56, 56]

        x = self.resNet.layer1(x)  # out = [N, 64, 56, 56]
        self.save_acti(x,"layer1")
        x = self.resNet.layer2(x)  # out = [N, 128, 28, 28]
        self.save_acti(x, "layer2")
        x = self.resNet.layer3(x)  # out = [N, 256, 14, 14]
        self.save_acti(x, "layer3")
        x = self.resNet.layer4(x)  # out = [N, 512, 7, 7]
        self.save_acti(x, "layer4")
        return x  # out = [N, 512, 1 ,1]

    def forward(self, x):
        out_sum = self.resNet.forward(x)
        gray = x[:,0:1]#to_grayscale(x)
        self.DNet.forward(gray.double())
        #in_opti = self.DNet.concat_layer_modulus()  # self.get_resnet_convs_out(x)

        return out_sum


if __name__ == "__main__":
    config = DNET_config(None)
    a = OptiCNN(config,nFilmLayer=10)
    print(f"OptiCNN={a}")
    pass

