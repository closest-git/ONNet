'''
@Author: Yingshi Chen

@Date: 2020-04-08 17:12:34
@
# Description: 
'''
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage
import os
import math
import hdf5storage
from enum import Enum
import re
from torchvision.transforms import transforms
import cv2
import numpy as np

def get_data_if_needed(data_path='./data/', url="https://ndownloader.figshare.com/articles/1512427/versions/5"):
    if os.path.isdir(data_path):
        #_arrange_brain_tumor_data(data_path)
        print("Data directory already exists. ",
              "if from some reason the data directory structure is wrong please remove the data dir and rerun this script")
        return
    filename = "all_data.zip"
    download_url(url, data_path, filename)
    unzip_all_files(data_path)
    _arrange_brain_tumor_data(data_path)

def convert_landmark_to_bounding_box(landmark):
    x_min = x_max = y_min = y_max = None
    for x, y in landmark:
        if x_min is None:
            x_min = x_max = x
            y_min = y_max = y
        else:
            x_min, x_max = min(x, x_min), max(x, x_max)
            y_min, y_max = min(y, y_min), max(y, y_max)
    return [int(x_min), int(x_max), int(y_min), int(y_max)]

class ClassesLabels(Enum):
    Meningioma = 1
    Glioma = 2
    Pituitary = 3

    def __len__(self):
        return 3

def normalize(x,  mean=470, std=None):
    mean_tansor = torch.ones_like(x) * mean
    x -= mean_tansor
    if std:
        x /= std
    return x

# https://github.com/galprz/brain-tumor-segmentation
class BrainTumorDataset(Dataset):    
    def __init__(self,config, root, train=True, download=True,
                                                  classes=(ClassesLabels.Meningioma,
                                                  ClassesLabels.Glioma,
                                                  ClassesLabels.Pituitary)):
        super().__init__()
        self.config = config
        test_fr = 0.15
        if download:
            get_data_if_needed(root)
        self.root = root
        # List all data files
        items = []
        if ClassesLabels.Meningioma in classes:
            items += ['meningioma/' + item for item in os.listdir(root + 'meningioma/')]
        if ClassesLabels.Glioma in classes:
            items += ['glioma/' + item for item in os.listdir(root + 'glioma/')]
        if ClassesLabels.Meningioma in classes:
            items += ['pituitary/' + item for item in os.listdir(root + 'pituitary/')]

        if train:
            self.items = items[0:math.floor((1-test_fr) * len(items)) + 1]
        else:
            self.items = items[math.floor((1-test_fr) * len(items)) + 1:]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if not (0 <= idx <  len(self.items)):
            raise IndexError("Idx out of bound")        
        if False:
            data = hdf5storage.loadmat(self.root + self.items[idx])['cjdata'][0]
            # transform the tumor border to array of (x, y) tuple
            xy = data[3]
            landmarks = []
            for i in range(0, len(xy), 2):
                x = xy[i][0]
                y = xy[i + 1][0]
                landmarks.append((x, y))
            mask = data[4]
            data[2].dtype = 'uint16'
            image = data[2] #ToPILImage()(data[2])
            image_with_metadata = {
                "label": int(data[0][0]),
                "image": image,
                "landmarks": landmarks,
                "mask": mask,
                "bounding_box": convert_landmark_to_bounding_box(landmarks)
            }
            return image_with_metadata
        else:
            return load_mat_trans(self.root + self.items[idx],target_size=self.config.IMG_size )    #(128,128)

def ToUint8(arr):
    a_0,a_1 = np.min(arr),np.max(arr)
    arr = (arr-a_0)/(a_1-a_0)*255
    arr = arr.astype(np.uint8)
    a_0,a_1 = np.min(arr),np.max(arr)
    return arr

def load_mat_trans(path,target_size=None):
    data_mat = hdf5storage.loadmat(path)
    data = data_mat['cjdata'][0]
    xy = data[3]
    landmarks = []
    for i in range(0, len(xy), 2):
        x = xy[i][0]
        y = xy[i + 1][0]
        landmarks.append((x, y))
    mask = data[4].astype(np.float32)
    m_0,m_1 = np.min(mask),np.max(mask)
    #data[2].dtype = 'uint16'
    image = data[2].astype(np.float32) #ToPILImage()(data[2])    
    if target_size is not None:
        image = cv2.resize(image,target_size)        
        #cv2.imshow("",image);           cv2.waitKey(0)          
        mask = cv2.resize(mask,target_size)  
        #cv2.imshow("",mask*255);         cv2.waitKey(0)  
        image = ToUint8(image)    
        mask = ToUint8(mask)    
    image_with_metadata = {
        "label": int(data[0][0]),
        "image": image,
        "landmarks": landmarks,
        "mask": mask,
        "bounding_box": convert_landmark_to_bounding_box(landmarks)
    }
    return image_with_metadata

mask_transformer = transforms.Compose([    
    transforms.ToTensor(),        
])

image_transformer_0 = transforms.Compose([   
    transforms.ToTensor(),    
    transforms.Lambda(lambda x: normalize(x))
])
image_transformer = transforms.Compose([   
    transforms.ToTensor(),    
])

class BrainTumorDatasetMask(BrainTumorDataset):
    def transform(self,image, mask):        
        img = image_transformer(image).float()
        mask = mask_transformer(mask).float()
        return img,mask

    def __init__(self,config, root, train=True, transform=None, classes=(ClassesLabels.Meningioma,
                                                  ClassesLabels.Glioma,
                                                  ClassesLabels.Pituitary)):
        super().__init__(config,root, train, classes=classes)
        #self.transform = brain_transform

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        sample = (item["image"], item["mask"])
        #return sample if self.transform is None else self.transform(*sample)
        img,mask = self.transform(item["image"], item["mask"])
        #i_0,i_1 = torch.min(img),torch.max(img)
        #m_0,m_1 = torch.min(mask),torch.max(mask)
        return img,mask

def _arrange_brain_tumor_data(root):
    # Remove and split files
    items = [item for item in filter(lambda item: re.search("^[0-9]+\.mat$", item), os.listdir(root))]
    try:
        os.mkdir(root + 'meningioma/')
    except:
        print("Meningioma directory already exists")
    try:
        os.mkdir(root + 'glioma/')
    except:
      print("Glioma directory already exists")
    try:
        os.mkdir(root + 'pituitary/')
    except:
        print("Pituitary directory already exists")

    for item in items:
        sample = hdf5storage.loadmat(root + item)['cjdata'][0]
        if sample[2].shape[0] == 512:
            if sample[0] == 1:
                os.rename(root + item, root + 'meningioma/' + item)
            if sample[0] == 2:
                os.rename(root + item, root + 'glioma/' + item)
            if sample[0] == 3:
                os.rename(root + item, root + 'pituitary/' + item)
        else:
            os.remove(root + item)

