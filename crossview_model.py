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

class _vgg(nn.Module):
    def __init__(self, feature_dim=512, is_vae=False):
        #512x7x7
        super(_vgg, self).__init__()

        self.is_vae = is_vae
        self.feature_dim = feature_dim

        self.extractor = models.vgg11_bn(pretrained=True)
        num_ftrs = self.extractor.classifier[0].in_features
        self.extractor.classifier = nn.Identity()

        self.mu = nn.Linear(num_ftrs, feature_dim)
        if is_vae:
            self.sigma = nn.Linear(num_ftrs, feature_dim)

    def forward(self, x):
        x = self.extractor(x)
        mu = self.mu(x)
        if self.is_vae:
            return mu, self.sigma(x)
        return mu

class _resnet(nn.Module):
    def __init__(self, feature_dim=512, is_vae=False):
        #512
        super(_resnet, self).__init__()

        self.is_vae = is_vae
        self.feature_dim = feature_dim

        self.extractor = models.resnet18(pretrained=True)
        num_ftrs = self.extractor.fc.in_features
        self.extractor.fc = nn.Identity()

        self.mu = nn.Linear(num_ftrs, feature_dim)
        if is_vae:
            self.sigma = nn.Linear(num_ftrs, feature_dim)

    def forward(self, x):
        x = self.extractor(x)
        mu = self.mu(x)
        if self.is_vae:
            return mu, self.sigma(x)
        return mu

class _densenet(nn.Module):
    def __init__(self, feature_dim=512, is_vae=False):
        #512
        super(_densenet, self).__init__()

        self.is_vae = is_vae
        self.feature_dim = feature_dim

        self.extractor = models.densenet169(pretrained=True)
        num_ftrs = self.extractor.classifier.in_features
        self.extractor.classifier = nn.Identity()

        self.mu = nn.Linear(num_ftrs, feature_dim)
        if is_vae:
            self.sigma = nn.Linear(num_ftrs, feature_dim)

    def forward(self, x):
        x = self.extractor(x)
        mu = self.mu(x)
        if self.is_vae:
            return mu, self.sigma(x)
        return mu

class ConvReLU(nn.Module):
    def __init__(self, inplanes, outplanes, ksize):
        super(ConvReLU, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, ksize, padding=ksize//2),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)

class Decoder(nn.Module):
    def __init__(self, inplanes=3, feature_dim=512):
        super(Decoder, self).__init__()
        
        self.inplanes = inplanes
        assert feature_dim >= 8
        self.feature_dim = feature_dim

        self.block = nn.Sequential(
            nn.Linear(feature_dim, feature_dim*7*7),
            nn.Unflatten(1, (feature_dim, 7, 7)),
            ConvReLU(feature_dim, feature_dim, 3),
            nn.ConvTranspose2d(feature_dim, feature_dim*2, kernel_size=2, stride=2),
            ConvReLU(feature_dim*2, feature_dim*2, 3),
            nn.ConvTranspose2d(feature_dim*2, feature_dim, kernel_size=2, stride=2),
            ConvReLU(feature_dim, feature_dim, 3),
            nn.ConvTranspose2d(feature_dim, feature_dim//2, kernel_size=2, stride=2),
            ConvReLU(feature_dim//2, feature_dim//2, 3),
            nn.ConvTranspose2d(feature_dim//2, feature_dim//4, kernel_size=2, stride=2),
            ConvReLU(feature_dim//4, feature_dim//4, 3),
            nn.ConvTranspose2d(feature_dim//4, feature_dim//8, kernel_size=2, stride=2),
            nn.Conv2d(feature_dim//8, inplanes, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)

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

class CrossViewModel(nn.Module):
    def __init__(self, backbone='vgg', inplanes=3, feature_dim=512, num_classes=11, is_vae=False, is_finetune=False):
        super(CrossViewModel, self).__init__()

        self.inplanes = inplanes
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.is_vae = is_vae
        
        assert backbone in ['vgg', 'resnet', 'densenet']
        self.backbone_type = backbone

        if backbone == 'vgg':
            self.backbone_a = _vgg(feature_dim, is_vae)
            self.backbone_g = _vgg(feature_dim, is_vae)
        elif backbone == 'resnet':
            self.backbone_a = _resnet(feature_dim, is_vae)
            self.backbone_g = _resnet(feature_dim, is_vae)
        elif backbone == 'densenet':
            self.backbone_a = _densenet(feature_dim, is_vae).extractor
            self.backbone_g = _densenet(feature_dim, is_vae)
        else:
            self.backbone_a = None
            self.backbone_g = None

        if is_finetune:
            for param in self.backbone_a.extractor.parameters():
                param.requires_grad = False
            for param in self.backbone_g.extractor.parameters():
                param.requires_grad = False

        self.decoder = Decoder(inplanes, 2*feature_dim)
        self.classifier = Classifier(2*feature_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        
        eps = torch.randn(mu.shape).to(mu.device)
        
        z = mu + eps*std
        
        return z

    def forward(self, a, g):
        if self.is_vae:
            mu_a, sigma_a = self.backbone_a(a)
            mu_g, sigma_g = self.backbone_g(g)

            feat_a = self.reparameterize(mu_a, sigma_a)
            feat_g = self.reparameterize(mu_g, sigma_g)
        else:
            feat_a = self.backbone_a(a)
            feat_g = self.backbone_g(g)
        
        joined_feats = torch.cat((feat_a, feat_g), 1)

        rec = self.decoder(joined_feats)
        c = self.classifier(joined_feats)

        if self.is_vae:
            return rec, c, mu_a, sigma_a, mu_g, sigma_g

        return rec, c

# Normal Distribution
class ELBOLoss(nn.Module):
    def __init__(self, rec_loss, alpha = 1):
        super(ELBOLoss, self).__init__()
        self.rec_loss = rec_loss
        self.alpha = alpha

    def forward(self, pred, target, mu_a, sigma_a, mu_g, sigma_g):
        kl_loss_a =  (-0.5*(1+sigma_a - mu_a**2- torch.exp(sigma_a)).sum(dim = 1)).mean(dim =0)
        kl_loss_g =  (-0.5*(1+sigma_g - mu_g**2- torch.exp(sigma_g)).sum(dim = 1)).mean(dim =0)
        rec_loss = self.rec_loss(pred, target)

        return rec_loss*self.alpha + kl_loss_a + kl_loss_g