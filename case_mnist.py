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
import torchvision
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

#dataset="emnist"
#dataset="fasion_mnist"
#dataset="cifar"
dataset="mnist"
# IMG_size = (28, 28)
# IMG_size = (56, 56)
IMG_size = (112, 112)
# IMG_size = (14, 14)
batch_size = 128

#net_type = "OptFormer"
#net_type = "cnn"
net_type = "DNet"
#net_type = "WNet"
#net_type = "MF_WNet"
#net_type = "MF_DNet";
#net_type = "BiDNet"

class Fasion_Net(nn.Module):        #https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Mnist_Net(nn.Module):
    def __init__(self,config, nCls=10):
        super(Mnist_Net, self).__init__()
        self.title = "Mnist_Net"
        self.config = config
        self.config.learning_rate = 0.01
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

train_trans = transforms.Compose([
        #transforms.RandomAffine(5,translate=(0,0.1)),
        #transforms.RandomRotation(10),
        #transforms.Grayscale(),
        transforms.Resize(IMG_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Convert a color image to grayscale and normalize the color range to [0,1].
        #transforms.Normalize((0.1307,), (0.3081,))
    ])
test_trans = transforms.Compose([
    #transforms.Grayscale(),
    transforms.Resize(IMG_size),
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
])

def train(model, device, train_loader, epoch, optical_trans,visual):
    #model.visual = visual
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.config.learning_rate, weight_decay=0.0005)
    if epoch==1:
        print(f"\n=======dataset={dataset} net={net_type} IMG_size={IMG_size} batch_size={batch_size}")
        print(f"======={model.config}")
        print(f"======={optimizer}")
        print(f"======={train_trans}\n")

    nClass = model.nClass
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx==0:   #check data_range
            d0,d1=data.min(),data.max()
            assert(d0>=0)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(optical_trans(data))
        #output = model(data)
        loss = model.loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            aLoss = loss.item()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader),aLoss ))
            #visual.UpdateLoss(title=f"Accuracy on \"{dataset}\"", legend=f"{model.legend()}", loss=aLoss, yLabel="Accuracy")
        #break

def test_one_batch(model,data,target,device):
    data, target = data.to(device), target.to(device)
    output = model(data)
    # output = model(data)
    loss = model.loss(output, target, reduction='sum').item()  # sum up batch loss
    pred = model.predict(output)
    # pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    return loss,correct

def test(model, device, test_loader, optical_trans,visual):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            loss, corr = test_one_batch(model, data, target, device)
            test_loss += loss
            correct += corr
            if False:
                data, target = data.to(device), target.to(device)
                if optical_trans is not None:       data = optical_trans(data)
                output = model(data)
                #output = model(data)
                test_loss += model.loss(output, target, reduction='sum').item() # sum up batch loss
                pred = model.predict(output)
                #pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accu = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, len(test_loader.dataset),accu))
    if visual is not None:
        visual.UpdateLoss(title=f"Accuracy on \"{dataset}\"",legend=f"{model.legend()}", loss=accu,yLabel="Accuracy")
    return accu

def Some_Test():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model_path = "E:/ONNet/checkpoint/DNNet_exp_W_H_Express Wavenet_[17,81.91]_.pth"
    PTH = torch.load(model_path)
    env_title, model = DNet_instance(PTH['net_type'], PTH['dataset'],
                                     PTH['IMG_size'], PTH['lr_base'], PTH['batch_size'], PTH['nClass'], PTH['nLayer'])
    epoch, acc = PTH['epoch'], PTH['acc']
    model.load_state_dict(PTH['net'])
    model.to(device)
    print(f"Load model@{model_path} epoch={epoch},acc={acc}")

    visual = Visdom_Visualizer(env_title,plots=[{"object":"output"}])
    visual.img_dir = "./dump/X_images/"
    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data', train=False,transform=test_trans),
        batch_size=batch_size, shuffle=False)
    if True:    #only one batch
        dataiter = iter(test_loader)
        images, target = dataiter.next()
        model.visual = visual
        loss,correct = test_one_batch(model, images, target, device)
        model.visual = None

    if False:
        acc_1 = test(model, device, test_loader, None, None)
        print(f"Some_Test acc={acc}-{acc_1}")

def main():
    #OnInitInstance()
    lr_base = 0.002
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

    nLayer = 10
    if dataset=="emnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('./data',split="balanced", train=True, download=True, transform=train_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(
            datasets.EMNIST('./data',split="balanced", train=False, transform=test_trans),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        # balanced=47       byclass=62
        nClass = 47
    elif dataset=="fasion_mnist":
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data',train=True, download=True, transform=train_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('./data',train=False, transform=test_trans),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        nClass = 10
    elif dataset=="cifar":
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data',train=True, download=True, transform=train_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(datasets.CIFAR10('./data',train=False, transform=test_trans),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        nClass = 10;        lr_base=0.005
    else:
        nClass = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,transform=train_trans),
            batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False,transform=test_trans),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    config_0 = NET_config(net_type,dataset,IMG_size,lr_base,batch_size,nClass,nLayer)
    env_title, model = DNet_instance(config_0)          #net_type,dataset,IMG_size,lr_base,batch_size,nClass,nLayer    
    visual = Visdom_Visualizer(env_title=env_title)
    # visual = Visualize(env_title=env_title)
    model.to(device)
    print(model)
    # visual.ShowModel(model,train_loader)

    if False:       # So strange in initialize
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, 2. / math.sqrt(n))
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 2. / math.sqrt(m.in_features))
                m.bias.data.zero_()

    nzParams = Net_dump(model)
    if False:
        nzParams=0
        for name, param in model.named_parameters():
            if param.requires_grad:
                nzParams+=param.nelement()
                print(f"\t{name}={param.nelement()}")
        print(f"========All parameters={nzParams}")

    acc,best_acc = 0,0
    accu_=[]
    for epoch in range(1, 33):
        if False:
            assert os.path.isdir('checkpoint')
            pth_path = f'./checkpoint/{model.title}_[{epoch},{acc}]_.pth'
            torch.save({'net': model.state_dict(), 'acc': acc, 'epoch': epoch,}, pth_path)
 
        if hasattr(model,'visualize'):
            model.visualize(visual, f"E[{epoch-1}")        
        train( model, device, train_loader, epoch, optical_trans,visual)
        acc = test(model, device, test_loader, optical_trans,visual)
        accu_.append(acc)
        if acc > best_acc:
            state = {
                'net_type':net_type,'dataset':dataset,'IMG_size':IMG_size,'lr_base':lr_base,
                'batch_size':batch_size,'nClass':nClass, 'nLayer':nLayer,
                'net': model.state_dict(), 'acc': acc,'epoch': epoch,
            }
            assert os.path.isdir('checkpoint')
            pth_path = f'./checkpoint/{model.title}_[{epoch},{acc}]_.pth'
            torch.save(state, pth_path)
            best_acc = acc
    print(f"\n=======\n=======accu_history={accu_}\n")

    #if args.save_model:
    #   torch.save(model.state_dict(), "mnist_onn.pt")

'''
    单衍射层测试算例
    1) PIL加载图片 2)DiffractiveLayer forward 3)plt显示
'''
def layer_test():
    from PIL import Image
    img = Image.open("E:/ONNet/data/MNIST/test_2.jpg")    
    img = train_trans(img)

    config=NET_config(net_type,dataset,IMG_size,0.01,32,10,5)
    config.modulation = 'phase'
    config.init_value = "random"
    config.rDrop = 0        #drop out
    layer = DiffractiveLayer(IMG_size[0],IMG_size[1],config)

    out = layer.forward(img.cuda())
    im_out = layer.z_modulus(out)
    im_out = im_out.squeeze().cpu().detach().numpy()

    fig, ax = plt.subplots()
    #plt.axis('off')
    plt.grid(b=None)
    im = ax.imshow(im_out, interpolation='nearest', cmap='coolwarm')
    title = f"{layer.__repr__()}"
    ax.set_title(title,fontsize=12)
    fig.colorbar(im, orientation='horizontal')
    plt.show()
    plt.close()

    print("!!!Good Luck!!!")

if __name__ == '__main__':
    #Some_Test()
    #layer_test()
    main()
