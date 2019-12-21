'''
    tensorboard --logdir=runs
    https://localhost:6006/

    ONNX export failed on ATen operator ifft because torch.onnx.symbolic.ifft does not exist
'''

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#import visdom
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()


class Visualize:
    def __init__(self,title="onnet"):
        self.writer = SummaryWriter(f'runs/{title}')

    def ShowModel(self,model,data_loader):
        '''
            tensorboar显示效果较差
        '''
        dataiter = iter(data_loader)
        images, labels = dataiter.next()
        if True:
            img_grid = torchvision.utils.make_grid(images)
            matplotlib_imshow(img_grid, one_channel=True)
            self.writer.add_image('four_fashion_mnist_images', img_grid)
            self.writer.close()
        image_1 = images[0:1,:,:,:]
        images = images.cuda()
        self.writer.add_graph(model,images )
        self.writer.close()


def PROJECTOR_test():
    """ ==================使用PROJECTOR对高维向量可视化====================
        https://blog.csdn.net/wsp_1138886114/article/details/87602112
        PROJECTOR的的原理是通过PCA，T-SNE等方法将高维向量投影到三维坐标系（降维度）。
        Embedding Projector从模型运行过程中保存的checkpoint文件中读取数据，
        默认使用主成分分析法（PCA）将高维数据投影到3D空间中，也可以通过设置设置选择T-SNE投影方法，
        这里做一个简单的展示。
    """
    log_dirs = "../../runs/projector/"
    BATCH_SIZE = 256
    EPOCHS = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(datasets.MNIST('../../data', train=True, download=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ])),
                              batch_size=BATCH_SIZE, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=BATCH_SIZE, shuffle=True)

    class ConvNet(nn.Module):
        def __init__(self):
            super().__init__()
            # 1,28x28
            self.conv1 = nn.Conv2d(1, 10, 5)  # 10, 24x24
            self.conv2 = nn.Conv2d(10, 20, 3)  # 128, 10x10
            self.fc1 = nn.Linear(20 * 10 * 10, 500)
            self.fc2 = nn.Linear(500, 10)

        def forward(self, x):
            in_size = x.size(0)
            out = self.conv1(x)  # 24
            out = F.relu(out)
            out = F.max_pool2d(out, 2, 2)  # 12
            out = self.conv2(out)  # 10
            out = F.relu(out)
            out = out.view(in_size, -1)
            out = self.fc1(out)
            out = F.relu(out)
            out = self.fc2(out)
            out = F.log_softmax(out, dim=1)
            return out

    model = ConvNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())

    def train(model, DEVICE, train_loader, optimizer, epoch):
        n_iter = 0
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if (batch_idx + 1) % 30 == 0:
                n_iter = n_iter + 1
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

                # 主要增加了一下内容
                out = torch.cat((output.data.cpu(), torch.ones(len(output), 1)), 1)  # 因为是投影到3D的空间，所以我们只需要3个维度
                with SummaryWriter(log_dir=log_dirs, comment='mnist') as writer:
                    # 使用add_embedding方法进行可视化展示
                    writer.add_embedding(
                        out,
                        metadata=target.data,
                        label_img=data.data,
                        global_step=n_iter)

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # 损失相加
                pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))

    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)

    # 保存模型
    torch.save(model.state_dict(), './pytorch_tensorboardX_03.pth')

if __name__ == '__main__':
    PROJECTOR_test()