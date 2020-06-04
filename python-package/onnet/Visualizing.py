'''
    python -m visdom.server
    http://localhost:8097
    <env_name>.json file present in your ~/.visdom directory.

    tensorboard --logdir=runs
    http://localhost:6006/      非常奇怪的出错

    ONNX export failed on ATen operator ifft because torch.onnx.symbolic.ifft does not exist
'''
import seaborn as sns;      sns.set()
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter
import visdom
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import cv2
from torchvision import datasets, transforms
from .Z_utils import COMPLEX_utils as Z

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Visualize:
    def __init__(self,env_title="onnet",plots=[], **kwargs):
        self.log_dir = f'runs/{env_title}'
        self.plots = plots
        self.loss_step = 0
        self.writer = None  #SummaryWriter(self.log_dir)
        self.img_dir="./dump/images/"
        self.dpi = 100

    #https://stackoverflow.com/questions/9662995/matplotlib-change-title-and-colorbar-text-and-tick-colors
    def MatPlot(self,arr, title=""):
        fig, ax = plt.subplots()
        #plt.axis('off')
        plt.grid(b=None)
        im = ax.imshow(arr, interpolation='nearest', cmap='coolwarm')
        fig.colorbar(im, orientation='horizontal')
        plt.savefig(f'{self.img_dir}{title}.jpg')
        #plt.show()
        plt.close()

    def fig2data(self,fig):
        fig.canvas.draw()
        if True:  # https://stackoverflow.com/questions/42603161/convert-an-image-shown-in-python-into-an-opencv-image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        else:
            w, h = fig.canvas.get_width_height()
            buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
            buf.shape = (w, h, 4)
            # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
            buf = np.roll(buf, 3, axis=2)
            return buf

    '''
            sns.heatmap 很难用，需用自定义，参见https://stackoverflow.com/questions/53248186/custom-ticks-for-seaborn-heatmap
    '''
    def HeatMap(self, data, file_name, params={},noAxis=True, cbar=True):
        title,isSave = file_name,True
        if 'save' in params:
            isSave = params['save']
        if 'title' in params:
            title = params['title']
        path = '{}{}_.jpg'.format(self.img_dir, file_name)
        sns.set(font_scale=3)
        s = max(data.shape[1] / self.dpi, data.shape[0] / self.dpi)
        # fig.set_size_inches(18.5, 10.5)
        cmap = 'coolwarm'  # "plasma"  #https://matplotlib.org/examples/color/colormaps_reference.html
        # cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
        if noAxis:  # tight samples for training(No text!!!)
            figsize = (s, s)
            fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)
            ax = sns.heatmap(data, ax=ax, cmap=cmap, cbar=False, xticklabels=False, yticklabels=False)
            fig.savefig(path, bbox_inches='tight', pad_inches=0,figsize=(20,10))
            if False:
                image = cv2.imread(path)
                # image = fig2data(ax.get_figure())      #会放大尺寸，难以理解
                if (len(title) > 0):
                    assert (image.shape == self.args.spp_image_shape)  # 必须固定一个尺寸
                cv2.imshow("",image);       cv2.waitKey(0)
            plt.close("all")
            return path
        else:  # for paper
            ticks = np.linspace(0, 1, 10)
            xlabels = [int(i) for i in np.linspace(0, 56, 10)]
            ylabels = xlabels
            figsize = (s * 10, s * 10)
            #fig, ax = plt.subplots(figsize=figsize, dpi=self.dpi)  # more concise than plt.figure:
            fig, ax = plt.subplots(dpi=self.dpi)
            ax.set_title(title)
            # cbar_kws={'label': 'Reflex', 'orientation': 'horizontal'}
            # sns.set(font_scale=0.2)
            #  cbar_kws={'label': 'Reflex', 'orientation': 'horizontal'} , center=0.6
            # ax = sns.heatmap(data, ax=ax, cmap=cmap,yticklabels=ylabels[::-1],xticklabels=xlabels)
            # cbar_kws = dict(ticks=np.linspace(0, 1, 10))
            ax = sns.heatmap(data, ax=ax, cmap=cmap,vmin=-1.1, vmax=1.1, cbar=cbar) #
            #plt.ylabel('Incident Angle');            plt.xlabel('Wavelength(nm)')
            if False:
                ax.set_xticklabels(xlabels);            ax.set_yticklabels(ylabels[::-1])
                y_limit = ax.get_ylim();
                x_limit = ax.get_xlim()
                ax.set_yticks(ticks * y_limit[0])
                ax.set_xticks(ticks * x_limit[1])
            else:
                plt.axis('off')
            if False:
                plt.show(block=True)

            image = self.fig2data(ax.get_figure())
            plt.close("all")
            #image_all = np.concatenate((img_0, img_1, img_diff), axis=1)
            #cv2.imshow("", image);    cv2.waitKey(0)
            if isSave:
                cv2.imwrite(path, image)
                return path
            else:
                return image

    plt.close("all")

    def ShowModel(self,model,data_loader):
        '''
            tensorboar显示效果较差
        '''
        dataiter = iter(data_loader)
        images, labels = dataiter.next()
        if images.shape[0]>32:
            images=images[0:32,...]
        if True:
            img_grid = torchvision.utils.make_grid(images)
            matplotlib_imshow(img_grid, one_channel=True)
            self.writer.add_image('one_batch', img_grid)
            self.writer.close()
        image_1 = images[0:1,:,:,:]
        if False:
            images = images.cuda()
            self.writer.add_graph(model,images )
            self.writer.close()

    def onX(self,X,title,nMostPic=64):
        shape = X.shape
        if Z.isComplex(X):
            #X = torch.cat([X[..., 0],X[..., 1]],0)
            X = Z.modulus(X)
            X = X.cpu()
        if shape[1]!=1:
            X = X.contiguous().view(shape[0]*shape[1],1,shape[-2],shape[-1]).cpu()
        if X.shape[0]>nMostPic:
            X=X[:nMostPic,...]
        img_grid = torchvision.utils.make_grid(X).detach().numpy()
        plt.axis('off');
        plt.grid(b=None)
        image_np = np.transpose(img_grid, (1, 2, 0))
        min_val,max_val = np.max(image_np),np.min(image_np)
        image_np = (image_np - min_val) / (max_val - min_val)
        if title is None:
            plt.imshow(image_np)
            plt.show()
        else:
            path = '{}{}_.jpg'.format(self.img_dir, title)
            plt.imsave(path, image_np)


    def image(self, file_name, img_, params={}):
        #np.random.rand(3, 512, 256),
        #self.MatPlot(img_.cpu().numpy(),title=name)

        result = self.HeatMap(img_.cpu().numpy(),file_name,params,noAxis=False)
        return result

    def UpdateLoss(self,title,legend,loss,yLabel='LOSS',global_step=None):
        tag = legend
        step = self.loss_step if global_step==None else global_step
        with SummaryWriter(log_dir=self.log_dir) as writer:
            writer.add_scalar(tag, loss, global_step=step)
        #self.writer.close()  # 执行close立即刷新，否则将每120秒自动刷新
        self.loss_step = self.loss_step+1

class  Visdom_Visualizer(Visualize):
    '''
    封装了visdom的基本操作
    '''

    def __init__(self,env_title,plots=[], **kwargs):
        super(Visdom_Visualizer, self).__init__(env_title,plots)
        try:
            self.viz = visdom.Visdom(env=env_title, **kwargs)
            assert self.viz.check_connection()
        except:
            self.viz = None

    def UpdateLoss(self, title,legend, loss, yLabel='LOSS',global_step=None):
        self.vis_plot( self.loss_step, loss, title,legend,yLabel)
        self.loss_step = self.loss_step + 1

    def vis_plot(self,epoch, loss_, title,legend,yLabel):
        if self.viz is None:
            return
        self.viz.line(X=torch.FloatTensor([epoch]), Y=torch.FloatTensor([loss_]), win='loss',
                 opts=dict(
                     legend=[legend],  # [config_.use_bn],
                     fillarea=False,
                     showlegend=True,
                     width=1600,
                     height=800,
                     xlabel='Epoch',
                     ylabel=yLabel,
                     # ytype='log',
                     title=title,
                     # marginleft=30,
                     # marginright=30,
                     # marginbottom=80,
                     # margintop=30,
                 ),
                 update='append' if epoch > 0 else None)

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    ''' 非常奇怪的出错
    def image(self, name, img_, **kwargs):

        assert self.viz.check_connection()
        self.vis.image(
            np.random.rand(3, 512, 256),
            opts=dict(title='Random image as jpg!', caption='How random as jpg.', jpgquality=50),
        )
        self.vis.image (img_.cpu().numpy(),
                        #win=(name),
                        opts=dict(title=name),
                        **kwargs
                        )
    '''

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)
        print(self.log_text)

    def __getattr__(self, name):
        return getattr(self.vis, name)

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