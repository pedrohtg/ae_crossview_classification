import os
import time
import random
import numpy as np
import argparse
import torch
import torchvision

from torch import nn
from torch import optim
from torch.nn import functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils import data
from torch.backends import cudnn

from torchvision import models
from torchvision import datasets
from torchvision import transforms

from skimage import io
from skimage import transform as T
from skimage.util import img_as_float

from sklearn import metrics

from matplotlib import pyplot as plt

class CoffeeDataset(Dataset):
    def __init__(self, imgfolder, maskfolder, img_list=None):
        assert os.path.exists(imgfolder)
        assert os.path.exists(maskfolder)

        self.imgfolder = imgfolder
        self.maskfolder = maskfolder

        self.imgs = []
        if img_list is None:
            for f in os.listdir(self.imgfolder):
                if os.path.exists(os.path.join(self.imgfolder, f)) and os.path.exists(os.path.join(self.maskfolder, f)):
                    self.imgs.append(f)
        else:
            assert type(img_list) == type(list())
            self.imgs = img_list

        self.size = len(self.imgs)

    def __getitem__(self, index):
        imgname = os.path.join(self.imgfolder, self.imgs[index])
        mskname = os.path.join(self.maskfolder, self.imgs[index])

        img = io.imread(imgname)
        msk = np.array(io.imread(mskname))
        msk[np.where(msk == 255)] = 1

        #img = np.moveaxis(img, -1, 0)
        img = np.moveaxis(img_as_float(img), -1, 0).astype(np.float32)
        #return np.array(img/np.max(img)).astype(np.float32), msk
        return img, msk

    def __len__(self):
        return self.size

class CoffeeDatasetModular(Dataset):
    def __init__(self, imgfolder, maskfolder, usable=1.0, img_list=None):
        random.seed(100)
        assert os.path.exists(imgfolder)
        assert os.path.exists(maskfolder)

        self.imgfolder = imgfolder
        self.maskfolder = maskfolder

        self.allimgs = []

        if img_list is None:
            for f in os.listdir(self.imgfolder):
                if os.path.exists(os.path.join(self.imgfolder, f)) and os.path.exists(os.path.join(self.maskfolder, f)):
                    self.allimgs.append(f)
        else:
            assert type(img_list) == type(list())
            self.allimgs = img_list

        self.size = int(usable*len(self.allimgs))
        self.imgs = random.sample(self.allimgs, self.size)


    def __getitem__(self, index):
        imgname = os.path.join(self.imgfolder, self.imgs[index])
        mskname = os.path.join(self.maskfolder, self.imgs[index])

        img = io.imread(imgname)
        msk = np.array(io.imread(mskname))
        msk[np.where(msk == 255)] = 1

        img = np.moveaxis(img_as_float(img), -1, 0).astype(np.float32)

        return img, msk

    def __len__(self):
        return self.size


class CoffeeDatasetModularMT(Dataset):
    def __init__(self, imgfolder, maskfolder, usable=1.0, img_list=None, target_img_list=None):
        random.seed(100)
        assert os.path.exists(imgfolder)
        assert os.path.exists(maskfolder)

        self.imgfolder = imgfolder
        self.maskfolder = maskfolder

        self.allimgs = []
        self.labels = []

        if img_list is None:
            for f in os.listdir(self.imgfolder):
                if os.path.exists(os.path.join(self.imgfolder, f)) and os.path.exists(os.path.join(self.maskfolder, f)):
                    self.allimgs.append(f)
        else:
            assert type(img_list) == type(list())
            self.allimgs = img_list


        if target_img_list is None:
            self.label_size = int(usable*len(self.allimgs))
            self.size = len(self.allimgs)
            self.label_idx = random.sample(range(self.size), self.label_size)
            self.has_labels = [0] * self.size
            for idx in self.label_idx:
                self.has_labels[idx] = 1

        else:
            assert type(target_img_list) == type(list())
            self.target_img_list = target_img_list

            self.label_size = int(usable*len(self.target_img_list))
            self.size = len(self.target_img_list)
            self.label_idx = random.sample(range(self.size), self.label_size)
            self.has_labels = [0] * self.size
            
            for idx in self.label_idx:
                self.has_labels[idx] = 1

            aux = [1] * len(self.allimgs)
            self.allimgs += target_img_list
            self.size = len(self.allimgs) 
            self.has_labels = aux + self.has_labels



    def __getitem__(self, index):
        imgname = os.path.join(self.imgfolder, self.allimgs[index])
        mskname = os.path.join(self.maskfolder, self.allimgs[index])

        img = io.imread(imgname)
        msk = np.array(io.imread(mskname))
        msk[np.where(msk == 255)] = 1

        img = np.moveaxis(img_as_float(img), -1, 0).astype(np.float32)

        return img, msk, self.has_labels[index]

    def __len__(self):
        return self.size

class MultiLabelDatasetModular(Dataset):
    def __init__(self, imgfolder, maskfolder, usable=1.0, img_list=None):
        random.seed(30)
        assert os.path.exists(imgfolder)
        assert os.path.exists(maskfolder)

        self.imgfolder = imgfolder
        self.maskfolder = maskfolder

        self.allimgs = []

        if img_list is None:
            for f in os.listdir(self.imgfolder):
                if os.path.exists(os.path.join(self.imgfolder, f)) and os.path.exists(os.path.join(self.maskfolder, f)):
                    self.allimgs.append(f)
        else:
            assert type(img_list) == type(list())
            self.allimgs = img_list

        self.size = int(usable*len(self.allimgs))
        self.imgs = random.sample(self.allimgs, self.size)


    def __getitem__(self, index):
        imgname = os.path.join(self.imgfolder, self.imgs[index])
        mskname = os.path.join(self.maskfolder, self.imgs[index])

        img = io.imread(imgname)
        msk = np.array(io.imread(mskname))

        img = np.moveaxis(img_as_float(img), -1, 0).astype(np.float32)

        return img, msk

    def __len__(self):
        return self.size