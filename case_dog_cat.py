'''
    1)  https://github.com/rdcolema/pytorch-image-classification/blob/master/pytorch_model.ipynb
        https://github.com/mukul54/A-Simple-Cat-vs-Dog-Classifier-in-Pytorch/blob/master/catVsDog.py
'''
# https://github.com/mukul54/A-Simple-Cat-vs-Dog-Classifier-in-Pytorch/blob/master/catVsDog.py

import numpy as np # Matrix Operations (Matlab of Python)
import pandas as pd # Work with Datasources
import matplotlib.pyplot as plt # Drawing Library
from PIL import Image
import torch # Like a numpy but we could work with GPU by pytorch library
import torch.nn as nn # Nural Network Implimented with pytorch
import torchvision # A library for work with pretrained model and datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import glob
import os

image_size = (100, 100)
image_row_size = image_size[0] * image_size[1]

if False:   #https://medium.com/predict/using-pytorch-for-kaggles-famous-dogs-vs-cats-challenge-part-1-preprocessing-and-training-407017e1a10c
    import shutil
    import re
    files = os.listdir(train_dir)
    # Move all train cat images to cats folder, dog images to dogs folder
    for f in files:
        catSearchObj = re.search("cat", f)
        dogSearchObj = re.search("dog", f)
        if catSearchObj:
            shutil.move(f'{train_dir}/{f}', train_cats_dir)
        elif dogSearchObj:
            shutil.move(f'{train_dir}/{f}', train_dogs_dir)
    pass

class CatDogDataset(Dataset):
    def __init__(self, path, transform=None):
        self.classes = ["cat","dog"]   #os.listdir(path)
        self.path = path    #[f"{path}/{className}" for className in self.classes]
        #self.file_list = [glob.glob(f"{x}/*") for x in self.path]
        self.transform = transform

        files = []
        for i, className in enumerate(self.classes):
            query = f"{self.path}{className}*"
            cls_list = glob.glob(query)
            print(f"{className}:n={len(cls_list)}")
            for fileName in cls_list:
                files.append([i, className, fileName])
        self.file_list = files
        files = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fileName = self.file_list[idx][2]
        classCategory = self.file_list[idx][0]
        im = Image.open(fileName)
        if self.transform:
            im = self.transform(im)
        return im.view(-1), classCategory

#mean = [0.485, 0.456, 0.406];          std  = [0.229, 0.224, 0.225]
mean = [0.485];                         std  = [0.229]
transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

path    = '../data/dog_cat/train/'
dataset = CatDogDataset(path, transform=transform)
if True:
    def imshow(source):
        plt.figure(figsize=(10,10))
        imt = (source.view(-1, image_size[0], image_size[0]))
        imt = imt.numpy().transpose([1,2,0])
        imt = (std * imt + mean).clip(0,1)
        plt.subplot(1,2,2)
        plt.imshow(imt.squeeze())
    imshow(dataset[0][0])
    imshow(dataset[2][0])
    imshow(dataset[6000][0])
    plt.show()

shuffle     = True
batch_size  = 64
num_workers = 0
dataloader  = DataLoader(dataset=dataset,
                         shuffle=shuffle,
                         batch_size=batch_size,
                         num_workers=num_workers)

class MyModel(torch.nn.Module):
    def __init__(self, in_feature):
        super(MyModel, self).__init__()
        self.fc1     = torch.nn.Linear(in_features=in_feature, out_features=500)
        self.fc2     = torch.nn.Linear(in_features=500, out_features=100)
        self.fc3     = torch.nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = F.relu( self.fc1(x) )
        x = F.relu( self.fc2(x) )
        x = F.softmax( self.fc3(x), dim=1)
        return x

model = MyModel(image_row_size)
print(model)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)

epochs   = 10
for epoch in range(epochs):
    for i, (X,Y) in enumerate(dataloader):
#         x, y = dataset[i]
        yhat = model(X)
        loss = criterion(yhat.view(-1), Y)
        break