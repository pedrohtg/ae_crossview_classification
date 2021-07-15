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

from blocks import ConvReLU, EncoderBlock, DecoderBlockS, DecoderBlockU

# -------------------------------------
# ------- Segnet Architecture ---------
# -------------------------------------
class Segnet(nn.Module):

    def __init__(self, inputsize=3, numblocks=4, kernelsize=3, nclasses=2, numchannels=64):
        super(Segnet, self).__init__()
        
        self.inputsize = inputsize
        self.kernelsize = kernelsize
        self.nclasses=nclasses
        self.numchannels = numchannels
        self.numblocks = numblocks
        
        # Encoder Part
        self.encoder = nn.ModuleList()
        self.encoder.append(EncoderBlock(inputsize, numchannels, kernelsize, nconvs=2, pooling=True, returnpool=True))
        for i in range(1, numblocks):
            # The first two blocks have only two convolutions
            if i < 2:
                self.encoder.append(EncoderBlock(numchannels, numchannels, kernelsize, nconvs=2, pooling=True, returnpool=True))
            else:
                self.encoder.append(EncoderBlock(numchannels, numchannels, kernelsize, nconvs=3, pooling=True, returnpool=True))
            
        # Decoder Part
        self.decoder = nn.ModuleList()
        for i in range(numblocks-1):
            # The last two blocks have two convolutions
            if i >= numblocks-2:
                self.decoder.append(DecoderBlockS(numchannels, numchannels, kernelsize, nconvs=2))
            else:
                self.decoder.append(DecoderBlockS(numchannels, numchannels, kernelsize, nconvs=3))
        self.decoder.append(DecoderBlockS(numchannels, nclasses, kernelsize, nconvs=2))


    def forward(self, x):
        for i, enc in enumerate(self.encoder):
            x, idx = enc(x)
            self.decoder[self.numblocks - (i + 1)].setidx(idx)

        for i, dec in enumerate(self.decoder):
            x = dec(x)

        return x

# -----------------------------------------------
# ------- Segnet MultiTask Architecture ---------
# -----------------------------------------------
class SegnetMT(nn.Module):

    def __init__(self, inputsize=3, numblocks=4, kernelsize=3, nclasses=2, numchannels=64):
        super(SegnetMT, self).__init__()
        
        self.inputsize = inputsize
        self.kernelsize = kernelsize
        self.nclasses=nclasses
        self.numchannels = numchannels
        self.numblocks = numblocks
        
        # Encoder Part
        self.encoder = nn.ModuleList()
        self.encoder.append(EncoderBlock(inputsize, numchannels, kernelsize, nconvs=2, pooling=True, returnpool=True))
        for i in range(1, numblocks):
            # The first two blocks have only two convolutions
            if i < 2:
                self.encoder.append(EncoderBlock(numchannels, numchannels, kernelsize, nconvs=2, pooling=True, returnpool=True))
            else:
                self.encoder.append(EncoderBlock(numchannels, numchannels, kernelsize, nconvs=3, pooling=True, returnpool=True))
            
        # Decoder Part
        self.decoder_rec = nn.ModuleList()
        for i in range(numblocks-1):
            # The last two blocks have two convolutions
            if i >= numblocks-2:
                self.decoder_rec.append(DecoderBlockS(numchannels, numchannels, kernelsize, nconvs=2))
            else:
                self.decoder_rec.append(DecoderBlockS(numchannels, numchannels, kernelsize, nconvs=3))
        self.decoder_rec.append(DecoderBlockS(numchannels, inputsize, kernelsize, nconvs=2))

        self.decoder_seg = nn.ModuleList()
        for i in range(numblocks-1):
            # The last two blocks have two convolutions
            if i >= numblocks-2:
                self.decoder_seg.append(DecoderBlockS(numchannels, numchannels, kernelsize, nconvs=2))
            else:
                self.decoder_seg.append(DecoderBlockS(numchannels, numchannels, kernelsize, nconvs=3))
        self.decoder_seg.append(DecoderBlockS(numchannels, inputsize, kernelsize, nconvs=2))


    def encode(self, x):
        for i, enc in enumerate(self.encoder):
            x, idx = enc(x)
            self.decoder_rec[self.numblocks - (i + 1)].setidx(idx)
            self.decoder_seg[self.numblocks - (i + 1)].setidx(idx)

        return x

    def reconstruct(self, x):
        for i, dec in enumerate(self.decoder_rec):
            x = dec(x)

        return x

    def segment(self, x):
        for i, dec in enumerate(self.decoder_seg):
            x = dec(x)

        return x

    def forward(self, x):
        x = self.encode(x)

        return self.reconstruct(x), self.segment(x)
        


# -------------------------------------
# -------- Unet Architecture ----------
# -------------------------------------
class Unet(nn.Module):
    
    def __init__(self, inputsize=3, nclasses=2, numblocks=4, initchannels=32, kernelsize=3):
        super(Unet, self).__init__()

        self.inputsize = inputsize
        self.nclasses = nclasses
        self.numblocks = numblocks
        self.initchannels = initchannels
        self.kernelsize = kernelsize

        # Encoder Part
        self.encoder = nn.ModuleList()
        self.encoder.append(EncoderBlock(inputsize, initchannels, kernelsize))
        self.encs = []
        lastchannels = initchannels
        for i in range(1, numblocks):
            inchannels = lastchannels
            outchannels = inchannels*2

            self.encoder.append(EncoderBlock(inchannels, outchannels, kernelsize))

            lastchannels = outchannels

        self.center = DecoderBlockU(lastchannels, 2*lastchannels, lastchannels, kernelsize)

        # Decoder Part
        self.decoder = nn.ModuleList()

        for i in range(numblocks-1):
            inchannels = 2*lastchannels # To comport the concatenation of the enc features
            mchannels = lastchannels
            outchannels = lastchannels//2
            
            self.decoder.append(DecoderBlockU(inchannels, mchannels, outchannels, kernelsize))
            
            lastchannels = outchannels

        self.decoder.append(EncoderBlock(2*lastchannels, lastchannels, kernelsize, nconvs=2, pooling=False))

        self.lastchannels = lastchannels
        self.score = nn.Conv2d(lastchannels, nclasses, kernel_size=1)


    def forward(self, x):
        self.encs = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            self.encs.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.center(x)

        for i, dec in enumerate(self.decoder):
            x = dec(torch.cat((self.encs[self.numblocks - i - 1], x), dim=1))

        return self.score(x)


# -------------------------------------
# --------- FCN Architecture ----------
# -------------------------------------
class FCN(nn.Module):
    
    def __init__(self, inputsize=3, nclasses=2, numblocks=4, initchannels=32, fc=1024, kernelsize=3):
        super(FCN, self).__init__()
        self.inputsize = inputsize
        self.nclasses = nclasses
        self.numblocks = numblocks
        self.initchannels = initchannels
        self.kernelsize = kernelsize
        self.fc = fc

        # Encoder Part
        self.encoder = nn.ModuleList()
        self.encoder.append(EncoderBlock(inputsize, initchannels, kernelsize, pooling=True))
        self.encs = []
        self.numftmaps = [initchannels]
        channels = initchannels
        for i in range(1, numblocks):
            self.encoder.append(EncoderBlock(channels, 2*channels, kernelsize, pooling=True))
            self.numftmaps.append(2*channels)
            channels = 2*channels
        
        self.fc1 = ConvReLU(self.numftmaps[-1], fc, kernelsize)
        self.fc2 = ConvReLU(fc, fc, 1)

        self.scores = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.scores.append(nn.Conv2d(fc, self.nclasses, 1))
        self.ups.append(nn.ConvTranspose2d(self.nclasses, self.nclasses, kernel_size=2, stride=2))
        for i in range(1, numblocks):
            b = numblocks - i - 1
            self.scores.append(nn.Conv2d(self.numftmaps[b], self.nclasses, 1))
            self.ups.append(nn.ConvTranspose2d(self.nclasses, self.nclasses, kernel_size=2, stride=2))

    def forward(self, x):
        self.encs = []
        for i, enc in enumerate(self.encoder):
            x = enc(x)
            self.encs.append(x)

        x = self.fc1(x)
        x = self.fc2(x)

        x = self.scores[0](x) # x ~ enc[-1]
        x = self.ups[0](x)

        for i in range(1, self.numblocks):
            b = self.numblocks - i - 1
            scr = self.scores[i](self.encs[b])
            x = self.ups[i](x + scr)
        
        return x

class Discriminator(nn.Module):
    def __init__(self, lspace=1024):
        super(Discriminator, self).__init__()
        self.lspace = lspace

        self.net = nn.Sequential(
            nn.Linear(self.lspace, 48),
            nn.ReLU(),
            nn.Linear(48, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("Torch Version:")
    print(torch.__version__)

    print("Segnet Network")
    net = Segnet()
    print(net)

    print("Unet Network")
    net = Unet()
    print(net)

    print("FCN Network")
    net = FCN()
    print(net)