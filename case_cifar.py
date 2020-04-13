'''
    Train CIFAR10 with PyTorch.
    https://github.com/kuangliu/pytorch-cifar

    https://medium.com/@wwwbbb8510/lessons-learned-from-reproducing-resnet-and-densenet-on-cifar-10-dataset-6e25b03328da
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import sys
import argparse
CNN_MODEL_root = os.path.dirname(os.path.abspath(__file__))+"/python-package"
sys.path.append(CNN_MODEL_root)
from cnn_models import *
ONNET_DIR = os.path.abspath("./python-package/")
sys.path.append(ONNET_DIR)  # To find local version of the onnet
from onnet import *
import sys
import time
import torch.nn as nn
import torch.nn.init as init


# The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images.
IMG_size = (32, 32)
#IMG_size = (128, 128)
isDNet = True

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

#_, term_width = os.popen('stty size', 'r').read().split()
term_width = 80
TOTAL_BAR_LENGTH = 25.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    if current < total-1:
        sys.stdout.write('\r')
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    if False:
        # Go back to the center of the bar.
        for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
            sys.stdout.write('\b')
        sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        pass#sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
def Init():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(IMG_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(IMG_size),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # Model
    print('==> Building model..')
    if isDNet:
        #config_0 = RGBO_CNN_config("RGBO_CNN", 'cifar_10', IMG_size, lr_base=args.lr, batch_size=128, nClass=10, nLayer=5)
        #env_title, net = RGBO_CNN_instance(config_0)
        config_0 = NET_config("DNet",'cifar_10',IMG_size,lr_base=args.lr,batch_size=128, nClass=10, nLayer=10)
        env_title, net = DNet_instance(config_0)  
        config_base = net.config
    else:
        config_0 = NET_config("OpticalNet", 'cifar_10', IMG_size, lr_base=args.lr, batch_size=128, nClass=10)
        # net = VGG('VGG19')
        #net = ResNet34();           env_title='ResNet34';       net.legend = 'ResNet34'
        net = OpticalNet34(config_0);        env_title = 'OpticalNet34';        net.legend = 'OpticalNet34'
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92();                  env_title='DPN92';       net.legend = 'DPN92'
        # net = DPN26();        env_title = 'DPN92';        net.legend = 'DPN92'
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0();   env_title='EfficientNetB0'
        #visual = Visdom_Visualizer(env_title=env_title)

    print(net)
    Net_dump(net)
    net = net.to(device)
    visual = Visdom_Visualizer(env_title=env_title)
    #if hasattr(net, 'DInput'):        net.DInput.visual = visual  # 看一看

    if device == 'cuda':
        pass
        #net = torch.nn.DataParallel(net)        #https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html
        #cudnn.benchmark = True     #结果会有扰动 https://zhuanlan.zhihu.com/p/73711222

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
#using SGD with scheduled learning rate much better than Adam
    #optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    return net,trainloader,testloader,optimizer,criterion,visual

# Training
def train(epoch,net,trainloader,optimizer,criterion):
    print('\nEpoch: %d' % epoch)
    if epoch==0:
        #print(f"\n=======dataset={dataset} net={net_type} IMG_size={IMG_size} batch_size={batch_size}")
        #print(f"======={net.config}")
        print(f"======={optimizer}")
        #print(f"======={train_trans}\n")
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()     #retain_graph=True
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        #break


def test(epoch,net,testloader,criterion,visual):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            #break

    # Save checkpoint.
    acc = 100.*correct/total
    legend = "resnet"#net.module.legend()
    visual.UpdateLoss(title=f"Accuracy on \"cifar_10\"", legend=f"{legend}", loss=acc, yLabel="Accuracy")
    if False and acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

if __name__ == '__main__':
    seed_everything(42)
    net,trainloader,testloader,optimizer,criterion,visual = Init()
    #legend = net.module.legend()

    for epoch in range(start_epoch, start_epoch+2000):
        train(epoch,net,trainloader,optimizer,criterion)
        test(epoch,net,testloader,criterion,visual)
