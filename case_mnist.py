from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import sys
ONNET_DIR = os.path.abspath("./python-package/")
sys.path.append(ONNET_DIR)  # To find local version of the onnet
from onnet import *

import math
nClass = 10
nLayer = 5
#dataset="emnist"
dataset="fasion_mnist"
#dataset="mnist"
#net_type = "cnn"
#net_type = "DNet"
net_type = "MultiDNet"
#net_type = "BiDNet"
IMG_size = (28, 28)
#IMG_size = (112, 112)
#IMG_size = (14, 14)
batch_size = 128

class BaseNet(nn.Module):
    def __init__(self, nCls=10):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.isDropOut = False
        self.nFC=1
        if self.isDropOut:
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
        if IMG_size[0]==56:
            nFC1 = 43264
        else:
            nFC1 = 9216
        if self.nFC == 1:
            self.fc1 = nn.Linear(nFC1, 10)
        else:
            self.fc1 = nn.Linear(nFC1, 128)
            self.fc2 = nn.Linear(128, 10)
        self.loss = F.cross_entropy
        self.nClass = nCls

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        if self.isDropOut:
            x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        if self.isDropOut:
            x = self.dropout2(x)
        if self.nFC == 2:
            x = self.fc2(x)
        #output = F.log_softmax(x, dim=1)
        output = x
        return output

    def predict(self,output):
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        return pred

class View(nn.Module):
    def __init__(self, *args):
        super(View, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(-1,*self.shape)

#_loss_ = UserLoss.cys_loss

def train(model, device, train_loader, epoch, optical_trans):
    print(f"\n=======dataset={dataset} net={net_type} IMG_size={IMG_size} batch_size={batch_size}")
    print(f"======={model.config}\n")

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate,  weight_decay=0.0005)

    nClass = model.nClass
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(optical_trans(data))
        #output = model(data)
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        #break

def test(model, device, test_loader, optical_trans):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(optical_trans(data))
            #output = model(data)
            test_loss += model.loss(output, target, reduction='sum').item() # sum up batch loss
            pred = model.predict(output)
            #pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    """Train a simple Hybrid Scattering + CNN model on MNIST.

        Three models are demoed:
        'linear' - optical_trans + linear model
        'mlp' - optical_trans + MLP
        'cnn' - optical_trans + CNN

        optical_trans 1st order can also be set by the mode
        Scattering features are normalized by batch normalization.

        scatter + linear achieves 99.15% in 15 epochs
        scatter + cnn achieves 99.3% in 15 epochs

    """

    parser = argparse.ArgumentParser(description='MNIST optical_trans  + hybrid examples')
    parser.add_argument('--mode', type=int, default=2,help='optical_trans 1st or 2nd order')
    parser.add_argument('--classifier', type=str, default='linear',help='classifier model')
    args = parser.parse_args()
    assert(args.classifier in ['linear','mlp','cnn'])

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    optical_trans = OpticalTrans()

    # DataLoaders
    if use_cuda:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = None
        pin_memory = False

    train_trans = transforms.Compose([
        transforms.RandomAffine(5),
        transforms.RandomRotation(10),
        transforms.Resize(IMG_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_trans = transforms.Compose([
        transforms.Resize(IMG_size),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if dataset=="emnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('./data',split="balanced", train=True, download=True, transform=train_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('../data',split="balanced", train=False, transform=test_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        # balanced=47       byclass=62
        nClass = 47
        nLayer = 20
    elif dataset=="fasion_mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data',train=True, download=True, transform=train_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data',train=False, transform=test_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        # balanced=47       byclass=62
        nClass = 10
        nLayer = 5
    else:
        nClass = 10
        nLayer = 5
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,transform=train_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False,transform=test_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    if net_type == "cnn":
        model = BaseNet()
    elif net_type == "DNet":
        model = D2NNet(IMG_size,nClass,nLayer,DNET_config(batch=batch_size))
        model.double()
    elif net_type == "MultiDNet":
        #model = MultiDNet(IMG_size, nClass, nLayer,[0.3e12,0.35e12,0.4e12,0.42e12,0.5e12,0.6e12], DNET_config())
        model = MultiDNet(IMG_size, nClass, nLayer, [0.3e12, 0.35e12, 0.4e12, 0.42e12], DNET_config(batch=batch_size))
        model.double()
    elif net_type == "BiDNet":
        model = D2NNet(IMG_size, nClass, nLayer, DNET_config(batch=batch_size,chunk="binary"))
        #model = D2NNet(IMG_size, nClass,nLayer, DNET_config(chunk="logit"))
        #model = BinaryDNet(IMG_size,nClass,nLayer,1)
        model.double()

    model.to(device)
    print(model)

    if False:       # So strange in initialize
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, 2. / math.sqrt(n))
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 2. / math.sqrt(m.in_features))
                m.bias.data.zero_()

    #
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"\t{name}={param.nelement()}")
    # Optimizer


    for epoch in range(1, 100):
        train( model, device, train_loader, epoch, optical_trans)
        test(model, device, test_loader, optical_trans)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_onn.pt")

if __name__ == '__main__':
    main()