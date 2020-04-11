'''
@Author: Yingshi Chen
@Date: 2020-04-06 15:50:21
@
# Description: 
'''
import numpy as np
import pandas as pd
import os
import random 
from shutil import copyfile
import pydicom as dicom
import cv2
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from PIL import Image
import logging
import sys 
import time
import datetime
import tqdm
import torch
from torch.optim import Adam
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score,classification_report
import glob
from typing import Callable, Any
from typing import NamedTuple, List
from case_brain import *
ONNET_DIR = os.path.abspath("./python-package/")
sys.path.append(ONNET_DIR)  # To find local version of the onnet
#sys.path.append(os.path.abspath("./python-package/cnn_models/")) 
from onnet import *
isONN=True
if not isONN:
    from cnn_models.resunet import DeepResUNet 
IMG_size=(128,128)

def train_transforms(config):
    width, height = config.IMG_size[0],config.IMG_size[1]
    trans_list = [
        transforms.Resize((height, width)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(degrees=20,
                                    translate=(0.15, 0.15),
                                    scale=(0.8, 1.2),
                                    shear=5)], p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3)], p=0.5),
        transforms.Grayscale(),
        transforms.ToTensor()
    ]
    return transforms.Compose(trans_list)

def val_transforms(config):
    width, height = config.IMG_size[0],config.IMG_size[1]
    trans_list = [
        transforms.Resize((height, width)),
        transforms.Grayscale(),
        transforms.ToTensor()
    ]
    return transforms.Compose(trans_list)

class LungMask_set(Dataset):
    def __init__(self, config, transforms,isTrain=True):
        self.config = config
        self.img_root = config.train_img_dir if isTrain else config.test_img_dir
        self.mask_root = config.train_mask_dir if isTrain else config.test_mask_dir
        self.img_trans = transforms
        self.msk_trans = transforms
        self.images = self.load_images(self.img_root)
        self.masks = self.load_images(self.mask_root)        

    def load_images(self, root_dir):
        files=[]
        query = f"{root_dir}*.jpg"
        files = glob.glob(query)        
        return files

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if False:
            img = Image.open(self.img_pths[idx]).convert("RGB")
            img_tensor = self.transforms(img)
            label = self.labels[idx]
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:
            img = Image.open(self.images[idx]).convert("RGB")
            mask = Image.open(self.masks[idx])               

        imag_1 = self.img_trans(img).float()
        mask_1 = self.msk_trans(mask).float()
        if False:
            m_0,m_1=np.min(mask_1),np.max(mask_1)
            mask_1 = mask_1>0     
            m_0,m_1=np.min(mask_1),np.max(mask_1)
        return imag_1,mask_1
  

def to_np(x):
    return x.data.cpu().numpy()

class BatchResult(NamedTuple):
    loss: float
    score: float


class EpochResult(NamedTuple):
    losses: List[float]
    score: float


class FitResult(NamedTuple):
    num_epochs: int
    train_loss: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]
    best_score: float

class Trainer:
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """
    def __init__(self,
                 model,
                 loss_fn,
                 optimizer,
                 objective_metric,
                 config,
                 tensorboard_logger=None,
                 tensorboard_log_images=True,
                 experiment_prefix=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        :param tensorboard_logger: tensordboard logger.
        """
        device = 'cuda' if config.gpu else 'cpu'      #"cuda"
        self.tensorboard_logger = tensorboard_logger

        if experiment_prefix is None:
            now = datetime.datetime.now()
            self.experiment_prefix = now.strftime("%Y-%m-%d\%H:%M:%S")
        else:
            self.experiment_prefix = experiment_prefix
        self.tensorboard_log_images = tensorboard_log_images
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.objective_metric = objective_metric
        self.device = device

        if self.device:
            model.to(self.device)

    def fit(self, dl_train: DataLoader, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_score = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs-1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            epoch_train_res = self.train_epoch(dl_train, verbose=verbose, **kw)
            train_loss.extend([float(x.item()) for x in epoch_train_res.losses])
            train_acc.append(float(epoch_train_res.score))

            epoch_test_res = self.test_epoch(dl_test, verbose=verbose, **kw)
            test_loss.extend([float(x.item()) for x in epoch_test_res.losses])
            test_acc.append(float(epoch_test_res.score))

            if best_score is None:
                best_score = epoch_test_res.score
            elif epoch_test_res.score > best_score:
                best_score = epoch_test_res.score
                if checkpoints is not None:
                    torch.save(self.model, checkpoints)
                    print("**** Checkpoint saved ****")
                epochs_without_improvement = 0
            else:
                if early_stopping is not None and epochs_without_improvement >= early_stopping:
                    print("Early stopping after %s with out improvement" % epochs_without_improvement)
                    break
                epochs_without_improvement += 1

            # ========================

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc, best_score)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train()  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.eval()  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    def train_batch(self, index, batch_data) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """

        X, y = batch_data
        if self.tensorboard_logger and self.tensorboard_log_images:
            B = torch.zeros_like(X.squeeze())
            C = torch.stack([B, X.squeeze(), X.squeeze()])
            C = C.unsqueeze(dim=0)
            images = C
            grid = make_grid(images, normalize=True, scale_each=True)
            self.tensorboard_logger.add_image("exp-%s/batch/test/images" % self.experiment_prefix, grid, index)
        if isinstance(X, tuple) or isinstance(X, list):
            X = [x.to(self.device) for x in X]
        else:
            X = X.to(self.device)
        y = y.to(self.device)
        pred = self.model(X)
        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        score = self.objective_metric(pred, y)
        if self.tensorboard_logger:
            self.tensorboard_logger.add_scalar('exp-%s/batch/train/loss' % self.experiment_prefix, loss, index)
            self.tensorboard_logger.add_scalar('exp-%s/batch/train/score' % self.experiment_prefix, score, index)
            if index % 300 == 0:
                for tag, value in self.model.named_parameters():
                    tag = tag.replace('.', '/')
                    self.tensorboard_logger.add_histogram('exp-%s/batch/train/param/%s' % (self.experiment_prefix, tag), to_np(value), index)
                    self.tensorboard_logger.add_histogram('exp-%s/batch/train/param/%s/grad' % (self.experiment_prefix, tag), to_np(value.grad), index)

        return BatchResult(loss, score)

    def test_batch(self, index, batch_data) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        with torch.no_grad():
            X, y = batch_data
            if isinstance(X, tuple) or isinstance(X, list):
                X = [x.to(self.device) for x in X]
            else:
                X = X.to(self.device)
            y = y.to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, y)
            score = self.objective_metric(pred, y)
            if self.tensorboard_logger:
                self.tensorboard_logger.add_scalar('exp-%s/batch/test/loss' % self.experiment_prefix, loss, index)
                self.tensorboard_logger.add_scalar('exp-%s/batch/test/score' % self.experiment_prefix, score, index)
            return BatchResult(loss, score)

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,file=pbar_file) as pbar:
            dl_iter = iter(dl)
            overall_score = overall_loss = avg_score = avg_loss = counter = 0
            min_loss = min_score = 1
            max_loss = max_score = 0
            for batch_idx in range(num_batches):
                counter += 1
                data = next(dl_iter)
                batch_res = forward_fn(batch_idx, data)
                if batch_res.loss > max_loss:
                    max_loss = batch_res.loss
                if batch_res.score > max_score:
                    max_score = batch_res.score

                if batch_res.loss < min_loss:
                    min_loss = batch_res.loss
                if batch_res.score < min_score:
                    min_score = batch_res.score
                overall_loss += batch_res.loss
                overall_score += batch_res.score
                losses.append(batch_res.loss)                
                avg_loss = overall_loss / counter
                avg_score = overall_score / counter
                pbar.set_description(f'{pbar_name} (Avg. loss:{avg_loss:.3f}, Avg. score:{avg_score:.3f})')                
                pbar.update()
                if counter%30==0:           print("")

            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, Min {min_loss:.3f}, Max {max_loss:.3f}), '
                                 f'(Avg. Score {avg_score:.4f}, Min {min_score:.4f}, Max {max_score:.4f})')

        return EpochResult(losses=losses, score=avg_score)

def UpdateConfig(config):
    config.random_seed = 42
    config.gpu = True
    config.batch_size = 4
    config.IMG_size =  IMG_size
    config.train_img_dir = "F:/Datasets/lung/fg/"
    config.train_mask_dir = "F:/Datasets/lung/alpha/"
    config.test_img_dir = "F:/Datasets/lung/fg/"
    config.test_mask_dir = "F:/Datasets/lung/alpha/"
    config.weights = None
    config.n_threads = 4
    #config.weights = "E:/Insegment/COVID-Next-Pytorch-master/COVIDNext50_NewData_F1_92.98_step_10800.pth"
    config.lr = 1e-4
    config.weight_decay = 1e-3
    config.lr_reduce_factor = 0.7
    config.lr_reduce_patience = 5
    config.epochs = 300
    config.log_steps = 5
    config.eval_steps = 400
    config.ckpts_dir = "./experiments/ckpts"
    return config

def dice_loss(pred, target, epsilon=1e-7, use_sigmoid=True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.sigmoid(pred)
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + epsilon) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)))
    return loss.mean()

def dice_coeff(pred, target, threshold=0.5, epsilon=1e-6, use_sigmoid = True):
    # make sure the tensors are align in memory and convert to probabilities if needed
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.sigmoid(pred)
    target = target.contiguous()

    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + epsilon) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)
    return dice.mean()


if __name__ == '__main__':
    #load_mat_test("F:/Datasets/brain/glioma/209.mat")
    config_0 = NET_config("WNet",'covid',IMG_size,0.01,batch_size=16, nClass=8, nLayer=20)
    #config_0 = RGBO_CNN_config("RGBO_CNN",'covid',IMG_size,0.01,batch_size=16, nClass=3, nLayer=5)
    if isONN:
        config_0.feat_extractor = "last_layer"
        env_title, net = DNet_instance(config_0)  
        #env_title, net = RGBO_CNN_instance(config_0)  
        config = net.config
        config = UpdateConfig(config)
        config.batch_size = 64
        config.log_steps = 10
        config.lr = 0.001
        state = None
    else:
        config = UpdateConfig(config_0)
        if config.weights:
            state = torch.load(config.weights)
            log.info("Loaded model weights from: {}".format(config.weights))
        else:
            state = None

        state_dict = state["state_dict"] if state else None
        net = DeepResUNet()
        if state_dict:
            net = load_model_weights(model=net, state_dict=state_dict,log=log)
    print(net)
    Net_dump(net)
    seed_everything(config.random_seed)    
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    criterion = dice_loss
    success_metric = dice_coeff
    trainer = Trainer(net, criterion, optimizer, dice_coeff, config, None)
#https://github.com/galprz/brain-tumor-segmentation/blob/master/experiment-DeepResUnet.ipynb

    if False:
        ds_train = LungMask_set(config,train_transforms(config))
        ds_test = LungMask_set( config,val_transforms(config),isTrain=False)
    else:
        config.batch_size = 16
        ds_train = BrainTumorDatasetMask(config,root="F:/Datasets/brain/", train=True)
        ds_test = BrainTumorDatasetMask(config,root="F:/Datasets/brain/", train=False)
    dl_train = torch.utils.data.DataLoader(ds_train, config.batch_size, shuffle=True)
    dl_test = torch.utils.data.DataLoader(ds_test, config.batch_size, shuffle=False)
    print(f"config={config}")
    fit_res = trainer.fit(dl_train,dl_test,num_epochs= config.epochs,checkpoints='dump/saved_models/' + net.__class__.__name__ + "V2")
