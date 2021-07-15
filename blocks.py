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

from sklearn import metrics

from matplotlib import pyplot as plt

# Convolutional block of encoder/ used in Unet, FCN, Segnet
class ConvReLU(nn.Module):
    def __init__(self, inp, out, ksize):
        super(ConvReLU, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(inp, out, ksize, padding=ksize//2),
            nn.BatchNorm2d(out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, ksize, nconvs=2, pooling=False, returnpool=False):
        super(EncoderBlock, self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=ksize
        self.nconvs = nconvs
        self.pooling = pooling
        self.returnpool = returnpool

        self.enc = nn.ModuleList()
        
        self.enc.append(ConvReLU(in_channels, out_channels, ksize))
        for i in range(1, nconvs):
            self.enc.append(ConvReLU(out_channels, out_channels, ksize))
        
        if pooling:
            self.maxp = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=returnpool)

    def forward(self, x):
        for i, e in enumerate(self.enc):
            x = e(x)

        if self.pooling:
            return self.maxp(x)
        else:
            return x

# Decoder Block for Unet
class DecoderBlockU(nn.Module):

    def __init__(self, in_channels, m_channels, out_channels, ksize, nconvs=2):
        super(DecoderBlockU, self).__init__()

        self.in_channels=in_channels
        self.m_channels = m_channels
        self.out_channels=out_channels
        self.kernel_size=ksize
        self.nconvs = nconvs

        self.dec = nn.ModuleList()
                
        self.dec.append(ConvReLU(in_channels, m_channels, ksize))
        for i in range(1, nconvs):
            self.dec.append(ConvReLU(m_channels, m_channels, ksize))
        self.dec.append(nn.ConvTranspose2d(m_channels, out_channels, kernel_size=2, stride=2))


    def forward(self, x):
        for i, d in enumerate(self.dec):
            x = d(x)
        return x

class DecoderBlockS(nn.Module):

    def __init__(self, in_channels, out_channels, ksize, nconvs=2):
        super(DecoderBlockS, self).__init__()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=ksize
        self.nconvs = nconvs

        self.dec = nn.ModuleList()
        self.idx = None
        
        self.dec.append(ConvReLU(in_channels, out_channels, ksize))
        for i in range(1, nconvs):
            self.dec.append(ConvReLU(out_channels, out_channels, ksize))

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def setidx(self, idx):
        self.idx = idx

    def forward(self, x):
        x = self.unpool(x, self.idx)
        for i, d in enumerate(self.dec):
            x = d(x)
        return x


if __name__ == "__main__":
    print(torch.__version__)
