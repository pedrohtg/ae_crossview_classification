import cv2
from scipy.spatial import distance
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
import torch.nn.functional as F
import time
import copy
import statistics
from tqdm import tqdm

def kmeans(k, descriptor_list):
    _kmeans = KMeans(n_clusters = k, n_init=10)
    _kmeans.fit(descriptor_list)
    return _kmeans

class Classifier(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=11):
        super(Classifier, self).__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.block = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, num_classes)
        )

    def forward(self, x):
        return self.block(x)


class BoFModel(nn.Module):
    def __init__(self, dataloaders, backbone='vgg', inplanes=3, feature_dim=512, num_classes=11, is_vae=False, is_finetune=False):
        super(BoFModel, self).__init__()

        self.inplanes = inplanes
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.is_vae = is_vae

        self.classifier = Classifier(2*feature_dim, num_classes)

        self.sift = cv2.xfeatures2d.SIFT_create()

        sift_vectors_a = {}
        descriptor_list_a = []

        sift_vectors_g = {}
        descriptor_list_g = []

        for data_a, data_g in tqdm(dataloaders['train']):
            inp_a = data_a[0][0]
            inp_g = data_g[0][0]

            inp_a = torch.movedim(inp_a, 1, 3)
            inp_g = torch.movedim(inp_g, 1, 3)

            for i in range(inp_a.size): 
                features = []
                for img in inp_a[i]:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    kp, des = self.sift.detectAndCompute(img, None)
            
                    descriptor_list_a.extend(des)
                    features.append(des)
                sift_vectors_a[i] = features

            for i in range(inp_g.size):
                features = []
                for img in inp_g[i]:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    kp, des = self.sift.detectAndCompute(img, None)
            
                    descriptor_list_g.extend(des)
                    features.append(des)
                sift_vectors_g[i] = features

        self.visual_words_a = kmeans(feature_dim, descriptor_list_a) 
        self.visual_words_g = kmeans(feature_dim, descriptor_list_g) 


    def forward(self, a, g):
        a = torch.movedim(a, 1, 3)
        g = torch.movedim(g, 1, 3)

        histo_list = []

        for i in range(a.size):
            for img_a, img_g in zip(a[i], g[i]):
                img_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2GRAY)
                kp_a, des_a = self.sift.detectAndCompute(img_a, None)

                histo_a = torch.zeros(self.feature_dim, device=self.classifier.device)
                nkp_a = len(kp_a)

                for d in des_a:
                    idx = self.visual_words_a.predict([d])
                    histo_a[idx] += 1/nkp_a
                
                img_g = cv2.cvtColor(img_g, cv2.COLOR_RGB2GRAY)
                kp_g, des_g = self.sift.detectAndCompute(img_g, None)

                histo_g = torch.zeros(self.feature_dim, device=self.classifier.device)
                nkp_g = len(kp_g)

                for d in des_g:
                    idx = self.visual_words_g.predict([d])
                    histo_g[idx] += 1/nkp_g

                histo_list.append(torch.cat([histo_a, histo_g]))
        
        histo_list = torch.vstack(histo_list)

        return a, g, self.classifier(histo_list)