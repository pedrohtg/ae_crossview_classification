import os
import time
import random
import numpy as np
import argparse
import torch
import torchvision
import copy

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

from net import Segnet, SegnetMT
from blocks import ConvReLU, EncoderBlock, DecoderBlockS, DecoderBlockU
from datasets import CoffeeDataset, CoffeeDatasetModular

import warnings
warnings.filterwarnings("ignore")

def trainAE(train_loader, net, criterion, optimizer, epoch, args=None):

    tic = time.time()
    
    # Setting network for training mode.
    net.train()

    # Average Meter for batch loss.
    train_loss = []

    # Lists for whole epoch loss.
    labs_all, prds_all = [], []

    # Iterating over batches.
    for i, batch_data in enumerate(train_loader):
        # Obtaining images and labels for batch.
        inps, labs = batch_data

        # Casting to cuda variables.
        inps = torch.Tensor(inps).to(args['device'])
        labs = torch.as_tensor(labs, device=args['device']).to(torch.int64)
        
        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = net(inps)
        prds = outs.data.cpu().numpy()

        #print(np.max(prds[0].flatten()))

        # Computing loss.
        loss = criterion(outs, inps)
        #print(loss)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()
        
        # Updating loss meter.
        train_loss.append(loss.data.item())

    toc = time.time()
    
    # Transforming list into numpy array.
    train_loss = np.asarray(train_loss)

    print('-------------------------------------------------------------------')
    print('[epoch %d], [train loss %.4f +/- %.4f], [time %.2f]' % (
        epoch, train_loss.mean(), train_loss.std(), (toc - tic)))
    print('-------------------------------------------------------------------')

    return train_loss.mean(), train_loss.std()

def trainVAE(train_loader, net, criterion, optimizer, epoch, args=None):
    tic = time.time()
    
    # Setting network for training mode.
    net.train()

    # Lists for losses and metrics.
    train_loss = []
    
    # Iterating over batches.
    for i, batch_data in enumerate(train_loader):

        # Obtaining images and labels for batch.
        inps, labs = batch_data
        
        # Casting to cuda variables and reshaping.
        inps = torch.Tensor(inps).to(args['device'])
        labs = torch.as_tensor(labs, device=args['device']).to(torch.int64)
        
        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs, mu, logvar = net(inps)

        # TO DO Computing total loss.
        loss_bce, loss_kld = criterion(outs, inps, mu, logvar)
        loss = loss_bce + loss_kld

        # Computing backpropagation.
        loss.backward()
        optimizer.step()
        
        # Updating lists.
        train_loss.append((loss_bce.data.item(),
                           args['lambda_var'] * loss_kld.data.item(),
                           loss.data.item()))
    
    toc = time.time()
    
    train_loss = np.asarray(train_loss)
    
    # Printing training epoch loss and metrics.
    print('-------------------------------------------------------------------')
    print('[epoch %d], [train bce loss %.4f +/- %.4f], [train kld loss %.4f +/- %.4f], [training time %.2f]' % (
        epoch, train_loss[:,0].mean(), train_loss[:,0].std(), train_loss[:,1].mean(), train_loss[:,1].std(), (toc - tic)))
    print('-------------------------------------------------------------------')

    return train_loss[:,2].mean(), train_loss[:, 2].std()

def trainAAE(train_loader, ae, discr, ae_criterion, dis_criterion, optimizer_ae, optimizer_dis, epoch, args=None):
    tic = time.time()
    
    # Setting network for training mode.
    ae.train()
    discr.train()

    # Average Meter for batch loss.
    train_loss_dis = []
    train_loss_rec = [] 

    # Lists for whole epoch loss.
    labs_all, prds_all = [], []

    #y_real = torch.ones(args['batch_size'], 1).to(args['device'])
    #y_fake = torch.zeros(args['batch_size'], 1).to(args['device'])

    # Iterating over batches.
    for i, batch_data in enumerate(train_loader):
        # Obtaining images and labels for batch.
        inps, labs = batch_data

        # Casting to cuda variables.
        inps = torch.Tensor(inps).to(args['device'])

        y_real = torch.ones(inps.shape[0], 1).to(args['device'])
        y_fake = torch.zeros(inps.shape[0], 1).to(args['device'])
        
        ##############
        # Opt Discri #
        ##############

        # Clears the gradients of optimizer.
        optimizer_dis.zero_grad()

        # Generate samples from Normal (0, I) 
        # Compute loss for real data
        z = torch.randn((inps.shape[0], ae.lspace)).to(args['device'])
        real_dis = discr(z)
        real_loss = dis_criterion(real_dis, y_real)
        
        # Compute loss for encoder generated data
        ae_out = ae.encode(inps)
        #print(ae_out.shape)
        fake_dis = discr(ae_out)
        #print(fake_dis.shape)
        fake_loss = dis_criterion(fake_dis, y_fake)

        dis_loss = real_loss + fake_loss

        dis_loss.backward()
        optimizer_dis.step()
        optimizer_ae.step()
        
        ###############
        # Opt Autoenc #
        ###############

        optimizer_ae.zero_grad()

        ae_rec = ae(inps)
        rec_loss = ae_criterion(ae_rec, inps)

        rec_loss.backward()
        optimizer_ae.step()


        train_loss_rec.append(rec_loss.data.item())
        train_loss_dis.append(dis_loss.data.item())


    toc = time.time()
    
    # Transforming list into numpy array.
    train_loss_rec = np.asarray(train_loss_rec)
    train_loss_dis = np.asarray(train_loss_dis)
    
    # Printing training epoch loss and metrics.
    print('-------------------------------------------------------------------')
    print('[epoch %d], [train rec loss %.4f +/- %.4f], [train dis loss %.4f +/- %.4f], [time %.2f]' % (
        epoch, train_loss_rec.mean(), train_loss_rec.std(), train_loss_dis.mean(), train_loss_dis.std(), (toc - tic)))
    print('-------------------------------------------------------------------')

    return train_loss_rec.mean(), train_loss_rec.std()

def trainNetwork(train_loader, net, criterion, optimizer, epoch, frozenblocks=2, segmentation=False, evaluation=None, args=None):

    tic = time.time()
    
    # Setting network for training mode and
    # freeze encoder blocks that were pre-trained
    net.train()
    for i in range(frozenblocks):
        net.encoder[i].eval()

    # Average Meter for batch loss.
    train_loss = []

    # Lists for whole epoch loss.
    labs_all, prds_all = [], []

    # Iterating over batches.
    for i, batch_data in enumerate(train_loader):

        # Obtaining images and labels for batch.
        inps, labs = batch_data

        # Casting to cuda variables.
        inps = torch.Tensor(inps).to(args['device'])
        #labs = torch.Tensor(labs).to(args['device']).to(torch.int64).squeeze()
        labs = torch.as_tensor(labs, device=args['device']).to(torch.int64)
        
        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        outs = net(inps)

        # Computing loss.
        if segmentation:
            loss = criterion(outs, labs)
        else:
            loss = criterion(outs, inps)

        # Obtaining predictions.
        if segmentation:
            prds = outs.data.max(1)[1].squeeze(1).cpu().numpy()
        else:
            prds = outs.data.cpu().numpy()

        # Appending images for epoch loss calculation.
        labs_all.append(labs.detach().data.squeeze(1).cpu().numpy())
        prds_all.append(prds)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

    toc = time.time()
    
    # Transforming list into numpy array.
    train_loss = np.asarray(train_loss)
    
    # Printing training epoch loss and metrics.
    # Computing error metrics for whole epoch.

    iou, normalized_acc = np.zeros(1), np.zeros(1)
    if epoch % args['pf'] == 0:
        if not (evaluation is None):
            iou, normalized_acc = evaluation(prds_all, labs_all) #evaluate(prds_all, labs_all, args['n_classes'])
            # Printing test epoch loss and metrics.
            print('-------------------------------------------------------------------')
            print('[epoch %d], [train loss %.4f +/- %.4f], [iou %.4f +/- %.4f], [normalized acc %.4f +/- %.4f], [time %.2f]' % (
                epoch, train_loss.mean(), train_loss.std(), iou.mean(), iou.std(), normalized_acc.mean(), normalized_acc.std(), (toc - tic)))
            print('-------------------------------------------------------------------')

        else:
            print('-------------------------------------------------------------------')
            print('[epoch %d], [train loss %.4f +/- %.4f], [time %.2f]' % (
                epoch, train_loss.mean(), train_loss.std(), (toc - tic)))
            print('-------------------------------------------------------------------')

    if not (evaluation is None):
        return 1.0-normalized_acc.mean(), normalized_acc.std()
    return train_loss.mean(), train_loss.std()

def test(test_loader, net, criterion, epoch, segmentation=False, evaluation=None, outputfolder=None, args=None):

    tic = time.time()
    
    # Setting network for evaluation mode.
    net.eval()

    # Average Meter for batch loss.
    test_loss = []

    # Lists for whole epoch loss.
    labs_all, prds_all = [], []

    # Iterating over batches.
    for i, batch_data in enumerate(test_loader):
        # Obtaining images and labels for batch.
        inps, labs = batch_data

        # Casting to cuda variables.
        inps = torch.Tensor(inps).to(args['device'])
        #labs = torch.Tensor(labs).to(args['device']).to(torch.int64).squeeze()
        labs = torch.as_tensor(labs, device=args['device']).to(torch.int64)
        
        # Forwarding.
        if args['aetype'] == 'vae' and segmentation == False:
            outs, mu, logvar = net(inps)
        else: 
            outs = net(inps)
        #print(outs.shape)
        
        # Computing loss.
        if segmentation:
            loss = criterion(outs, labs)
        elif args['aetype'] == 'vae' and segmentation == False:
            lossbce, losskld = criterion(outs, inps, mu, logvar)
            loss = lossbce + losskld
        else:
            loss = criterion(outs, inps)

        # Obtaining predictions.
        if segmentation:
            prds = outs.data.max(1)[1].squeeze(1).cpu().numpy()
        else:
            prds = outs.data.cpu().numpy()

        # Appending images for epoch loss calculation.
        labs_all.append(labs.detach().data.squeeze(1).cpu().numpy())
        prds_all.append(prds)

        # Updating loss meter.
        test_loss.append(loss.data.item())
    
    toc = time.time()
    
    # Transforming list into numpy array.
    test_loss = np.asarray(test_loss)
    
    # Computing error metrics for whole epoch.
    iou, normalized_acc = np.zeros(1), np.zeros(1)
    if epoch % args['pf'] == 0:
        if not (evaluation is None):
            iou, normalized_acc = evaluation(prds_all, labs_all)
            # Printing test epoch loss and metrics.
            print('-------------------------------------------------------------------')
            print('[epoch %d], [test loss %.4f +/- %.4f], [iou %.4f +/- %.4f], [normalized acc %.4f +/- %.4f], [time %.2f]' % (
                epoch, test_loss.mean(), test_loss.std(), iou.mean(), iou.std(), normalized_acc.mean(), normalized_acc.std(), (toc - tic)))
            print('-------------------------------------------------------------------')

        else:
            print('-------------------------------------------------------------------')
            print('[epoch %d], [test loss %.4f +/- %.4f], [time %.2f]' % (
                epoch, test_loss.mean(), test_loss.std(), (toc - tic)))
            print('-------------------------------------------------------------------')


    if not outputfolder is None:
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)
        #print(len(prds_all))
        for i, img in enumerate(prds_all[0]):
            #print("SHAPE", img.shape)
            name = ""
            if segmentation:
                name = 'seg_'
                name += "pred_" + str(i) + '_epoch_' + str(epoch) + '.png'

                io.imsave(os.path.join(outputfolder,name), img.astype('uint8')*255)
            else:
                name = 'rec_'
                name += "pred_" + str(i) + '_epoch_' + str(epoch) + '.png'
                sv = np.clip(np.moveaxis(img, 0, -1), 0, 1.01) 
                #sv = np.clip((np.abs(np.min(img)) + np.moveaxis(img, 0, -1))/np.max(img + np.abs(np.min(img))+0.0001), 0, 1.1)
                io.imsave(os.path.join(outputfolder,name), sv)

    if not (evaluation is None):
        return 1.0-normalized_acc.mean(), normalized_acc.std()
    return test_loss.mean(), test_loss.std()



def trainMT(train_loader, net, rec_criterion, seg_criterion, optimizer, epoch, frozenblocks=0, evaluation=None, args=None):

    tic = time.time()
    
    # Setting network for training mode and
    # freeze encoder blocks that were pre-trained
    net.train()
    for i in range(frozenblocks):
        net.encoder[i].eval()

    # Average Meter for batch loss.
    train_loss = []

    # Lists for whole epoch loss.
    labs_all, recs_all, segs_all = [], [], []

    # Iterating over batches.
    for i, batch_data in enumerate(train_loader):

        # Obtaining images and labels for batch.
        inps, labs, has_labels = batch_data

        # Casting to cuda variables.
        inps = torch.Tensor(inps).to(args['device'])
        #labs = torch.Tensor(labs).to(args['device']).to(torch.int64).squeeze()
        labs = torch.as_tensor(labs, device=args['device']).to(torch.int64)
        has_labels = torch.tensor(has_labels, dtype=torch.bool).to(args['device'])
        
        # Clears the gradients of optimizer.
        optimizer.zero_grad()

        # Forwarding.
        recs, segs = net(inps)

        total_labels = torch.sum(has_labels, dtype=int)

        loss = args['rec_wt']*rec_criterion(recs, inps)

        if total_labels > 0:
            pre_seg_loss = seg_criterion(segs, labs)
            seg_loss = torch.mean(pre_seg_loss[has_labels])

            loss += (1-args['rec_wt'])*seg_loss

        # Obtaining predictions.
        
        prds_seg = segs.data.max(1)[1].squeeze(1).cpu().numpy()
        prds_rec = recs.data.cpu().numpy()

        # Appending images for epoch loss calculation.
        labs_all.append(labs.detach().data.squeeze(1).cpu().numpy())
        recs_all.append(prds_rec)
        segs_all.append(prds_seg)

        # Computing backpropagation.
        loss.backward()
        optimizer.step()

        # Updating loss meter.
        train_loss.append(loss.data.item())

    toc = time.time()
    
    # Transforming list into numpy array.
    train_loss = np.asarray(train_loss)
    
    # Printing training epoch loss and metrics.
    # Computing error metrics for whole epoch.

    iou, normalized_acc = np.zeros(1), np.zeros(1)
    if epoch % args['pf'] == 0:
        if not (evaluation is None):
            iou, normalized_acc = evaluation(prds_all, labs_all) #evaluate(prds_all, labs_all, args['n_classes'])
            # Printing test epoch loss and metrics.
            print('-------------------------------------------------------------------')
            print('[epoch %d], [train loss %.4f +/- %.4f], [iou %.4f +/- %.4f], [normalized acc %.4f +/- %.4f], [time %.2f]' % (
                epoch, train_loss.mean(), train_loss.std(), iou.mean(), iou.std(), normalized_acc.mean(), normalized_acc.std(), (toc - tic)))
            print('-------------------------------------------------------------------')

        else:
            print('-------------------------------------------------------------------')
            print('[epoch %d], [train loss %.4f +/- %.4f], [time %.2f]' % (
                epoch, train_loss.mean(), train_loss.std(), (toc - tic)))
            print('-------------------------------------------------------------------')

    if not (evaluation is None):
        return 1.0-normalized_acc.mean(), normalized_acc.std()
    return train_loss.mean(), train_loss.std()

def testMT(test_loader, net, rec_criterion, seg_criterion, epoch, segmentation=False, evaluation=None, outputfolder=None, args=None):

    tic = time.time()
    
    # Setting network for evaluation mode.
    net.eval()

    # Average Meter for batch loss.
    test_loss = []

    # Lists for whole epoch loss.
    labs_all, recs_all, segs_all = [], [], []

    # Iterating over batches.
    for i, batch_data in enumerate(test_loader):
        # Obtaining images and labels for batch.
        # Obtaining images and labels for batch.
        inps, labs, has_labels = batch_data

        # Casting to cuda variables.
        inps = torch.Tensor(inps).to(args['device'])
        #labs = torch.Tensor(labs).to(args['device']).to(torch.int64).squeeze()
        labs = torch.as_tensor(labs, device=args['device']).to(torch.int64)
        has_labels = torch.tensor(has_labels, dtype=torch.bool).to(args['device'])
        

        # Forwarding.
        recs, segs = net(inps)

        total_labels = torch.sum(has_labels, dtype=int)

        loss = args['rec_wt']*rec_criterion(recs, inps)

        if total_labels > 0:
            pre_seg_loss = seg_criterion(segs, labs)
            seg_loss = torch.mean(pre_seg_loss[has_labels])

            loss += (1-args['rec_wt'])*seg_loss

        # Obtaining predictions.
        
        prds_seg = segs.data.max(1)[1].squeeze(1).cpu().numpy()
        prds_rec = recs.data.cpu().numpy()

        # Appending images for epoch loss calculation.
        labs_all.append(labs.detach().data.squeeze(1).cpu().numpy())
        recs_all.append(prds_rec)
        segs_all.append(prds_seg)

        # Updating loss meter.
        test_loss.append(loss.data.item())
    
    toc = time.time()
    
    # Transforming list into numpy array.
    test_loss = np.asarray(test_loss)
    
    # Computing error metrics for whole epoch.
    iou, normalized_acc = np.zeros(1), np.zeros(1)
    if epoch % args['pf'] == 0:
        if not (evaluation is None):
            iou, normalized_acc = evaluation(segs_all, labs_all)
            # Printing test epoch loss and metrics.
            print('-------------------------------------------------------------------')
            print('[epoch %d], [test loss %.4f +/- %.4f], [iou %.4f +/- %.4f], [normalized acc %.4f +/- %.4f], [time %.2f]' % (
                epoch, test_loss.mean(), test_loss.std(), iou.mean(), iou.std(), normalized_acc.mean(), normalized_acc.std(), (toc - tic)))
            print('-------------------------------------------------------------------')

        else:
            print('-------------------------------------------------------------------')
            print('[epoch %d], [test loss %.4f +/- %.4f], [time %.2f]' % (
                epoch, test_loss.mean(), test_loss.std(), (toc - tic)))
            print('-------------------------------------------------------------------')


    if not outputfolder is None:
        if not os.path.exists(outputfolder):
            os.mkdir(outputfolder)
        #print(len(prds_all))
        for i, img in enumerate(segs_all[0]):
            #print("SHAPE", img.shape)
        
            name = 'seg_'
            name += "pred_" + str(i) + '_epoch_' + str(epoch) + '.png'

            io.imsave(os.path.join(outputfolder,name), img.astype('uint8')*255)

            name = 'rec_'
            name += "pred_" + str(i) + '_epoch_' + str(epoch) + '.png'
            sv = np.clip(np.moveaxis(recs_all[0][i], 0, -1), 0, 1.01) 
            #sv = np.clip((np.abs(np.min(img)) + np.moveaxis(img, 0, -1))/np.max(img + np.abs(np.min(img))+0.0001), 0, 1.1)
            io.imsave(os.path.join(outputfolder,name), sv)

    # if not (evaluation is None):
    #     return 1.0-normalized_acc.mean(), normalized_acc.std()
    # return test_loss.mean(), test_loss.std()
    return test_loss.mean(), normalized_acc.mean()
