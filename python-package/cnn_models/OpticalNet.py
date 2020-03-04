import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
#ONNET_DIR = os.path.abspath("../../")
sys.path.append("../")  # To find local version of the onnet
from onnet import *
from onnet import DiffractiveLayer

class OpticalBlock(nn.Module):
    expansion = 1

    def __init__(self,config, in_planes, planes, stride=1):
        super(OpticalBlock, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        M,N = self.config.IMG_size[0], self.config.IMG_size[1]
        self.diffrac = DiffractiveLayer(M,N,config)
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        #assert x.shape[-1]==32 and x.shape[-2]==32
        #out += self.diffrac(x)
        out = F.relu(out)
        return out


class OpticalNet(nn.Module):
    def __init__(self, config,block, num_blocks):
        super(OpticalNet, self).__init__()
        num_classes = config.nClass
        self.config = config
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.config,self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def OpticalNet18(config):
    return OpticalNet(config,OpticalBlock, [2,2,2,2])

def OpticalNet34(config):
    return OpticalNet(config,OpticalBlock, [3,4,6,3])

def test():
    net = OpticalNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
