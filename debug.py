import argparse
import os, random
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from crossview_model import CrossViewModel, ELBOLoss
import dataloader
import trainval
import torch.nn as nn
import torch
import numpy as np

# Fix seed's for reproducibility
random.seed(13)
torch.manual_seed(13)

# Fixed image size
img_size = 224

#Dataset statistics
aerial_airound_mean, aerial_airound_std = [0.359, 0.382, 0.326], [0.251, 0.25, 0.24]
ground_airound_mean, ground_airound_std = [0.462, 0.49, 0.482], [0.26, 0.26, 0.267]
aerial_cvbrct_mean, aerial_cvbrct_std = [0.452, 0.442, 0.429], [0.256, 0.254, 0.253]
ground_cvbrct_mean, ground_cvbrct_std = [0.492, 0.497, 0.482], [0.257, 0.257, 0.26]

mean = {'aerial': {'airound': aerial_airound_mean, 'cvbrct': aerial_cvbrct_mean},
        'ground': {'airound': ground_airound_mean, 'cvbrct': ground_cvbrct_mean}}

std = {'aerial': {'airound': aerial_airound_std, 'cvbrct': aerial_cvbrct_std},
       'ground': {'airound': ground_airound_std, 'cvbrct': ground_cvbrct_std}}

def main():
    parser = argparse.ArgumentParser(description='CrossView classification')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset for normalization. [airound|cvbrct]')
    parser.add_argument('--aerial_dataset_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--ground_dataset_path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to folder where models and stats will be saved')
    parser.add_argument('--batch', type=int, required=True,
                        help='Batch Size')
    parser.add_argument('--epochs', type=int, required=True,
                        help='Number of epochs')
    parser.add_argument('--backbone', type=str, required=False, default='vgg',
                        help = 'Choose backbone network. [vgg|resnet|densenet]')
    parser.add_argument('--latent_dim', type=int, required=False, default=512,
                        help = 'Latent space dimensions')
    parser.add_argument('--network_type', type=str, required=False, default='ae',
                        help = 'Choose network type. [ae|vae]')
    parser.add_argument('--early_stop', type=int, required=True,
                        help='Number of epochs to activate early stop.')
    parser.add_argument('--feature_extract', type= bool, required=False, default=False,
                        help='Train just the classifier.')
    
    parser.add_argument('--optim', type=str, required=False, default='adam',
                        help='Optimizer used [adam|sgd].')
    parser.add_argument('--lr', type=float, required=False, default=0.001,
                        help='Learning Rate.')
    parser.add_argument('--momentum', type=float, required=False, default=0.9,
                        help='Learning Rate.')
    parser.add_argument('--wd', type=float, required=False, default=5e-5,
                        help='Weight Decay.')


    args = parser.parse_args()
    dataset = args.dataset
    aerial_dataset_path = args.aerial_dataset_path
    ground_dataset_path = args.ground_dataset_path
    out_dir = args.output_path
    batch_size = args.batch
    epochs = args.epochs
    net_type = args.network_type
    early_stop = args.early_stop
    feature_extract = args.feature_extract
    total_classes = len(os.listdir(os.path.join(aerial_dataset_path, 'train')))
    backbone = args.backbone
    latent_dim = args.latent_dim
    optim_type = args.optim
    lr = args.lr
    momentum = args.momentum
    wd = args.wd

    if (net_type == 'vae'):
        is_vae = True
    else:
        is_vae = False

    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)

    print ('.......Creating model.......')
    print('total classes: ', total_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CrossViewModel(backbone=backbone, inplanes=3, feature_dim=latent_dim, num_classes=total_classes, is_vae=is_vae, is_finetune=feature_extract)
    print (model)
    model = model.to(device)
    print ('......Model created.......')

    print ('......Creating dataloader......')
    dataloaders_dict = dataloader.create_dataloader(aerial_dataset_path, ground_dataset_path, img_size, batch_size,
                                                   mean['aerial'][dataset], std['aerial'][dataset],
                                                   mean['ground'][dataset], std['ground'][dataset])
    print ('......Dataloader created......')


    params_to_update = model.parameters()
    print("Params to learn:")
    print (params_to_update)

    if optim_type == 'sgd':
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9, weight_decay=wd)
    elif optim_type == 'adam':
        optimizer = optim.Adam(params_to_update, lr=lr, betas=(momentum, 0.99), weight_decay=wd)
    else:
        print("Optimizer Not Implemented.")
        exit()

    if is_vae:
        criterion = (ELBOLoss(nn.MSELoss()),nn.CrossEntropyLoss())
    else:
        criterion = (nn.MSELoss(), nn.CrossEntropyLoss())

    print("-"*30)
    print("Testing Model output ")
    inp = torch.randn(1, 3, 224, 224).to(device)
    labels = torch.ones(1, dtype=torch.long).to(device)
    if is_vae:
        rec_a, rec_g, clf, *aux_outputs = model(inp, inp)
        rec_loss = criterion[0]((rec_a, rec_g), (inp, inp), *aux_outputs)
        clf_loss = criterion[1](clf, labels)
        loss = 1*rec_loss + 1*clf_loss
    else:
        # Get model outputs and calculate loss
        rec_a, rec_g, clf = model(inp, inp)
        rec_loss = criterion[0]((rec_a, rec_g), (inp, inp))
        clf_loss = criterion[1](clf, labels)
        loss = 1*rec_loss + 1*clf_loss

    print(rec_a.shape, rec_g.shape, clf.shape)
    print(loss)
    loss.backward()

    print("-"*30)
    print("Testing Dataloader format ")
    for data_a, data_g in dataloaders_dict['train']:

        print(data_a)
        print('%'*30)
        print(len(data_a))
        inp_a, lab_a, img_a_paths = data_a[0], data_a[1], data_a[2]
        print(inp_a, lab_a, img_a_paths)
        print(data_g)

        break

    # tensor_board = SummaryWriter(log_dir = out_dir)
    # final_model, val_history = trainval.train(model, dataloaders_dict, criterion, optimizer,
    #                                          epochs, early_stop, tensor_board, is_vae)
    # print (out_dir)
    # if feature_extract:
    #     torch.save(final_model, os.path.join(out_dir, net_type + '_final_model_ft'))
    #     final_stats_file = open (os.path.join(out_dir, net_type + '_finalstats_ft.txt'), 'w')
    # else:
    #     torch.save(final_model, os.path.join(out_dir, net_type + '_final_model'))
    #     final_stats_file = open (os.path.join(out_dir, net_type + '_finalstats.txt'), 'w')
    # csv_file = open(os.path.join(out_dir, net_type + '_results.csv'), 'w')

    # trainval.final_eval(final_model, dataloaders_dict, csv_file, final_stats_file, is_vae)

if __name__ == '__main__':
    main()


#python debug.py --dataset airound --aerial_dataset_path /home/users/gabriel/Mestrado/datasets/airound_folds2/aerial_fold1/ --ground_dataset_path /home/users/gabriel/Mestrado/datasets/airound_folds2/ground_fold1/ --output_path ../tp_out --batch 10 --epochs 15 --backbone vgg --latent_dim 256 --network_type vae --early_stop 5