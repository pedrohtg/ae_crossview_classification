from torchvision import datasets, transforms
import os
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MultiImageFolder(datasets.ImageFolder):
    def __init__(self, aerial_ds, ground_ds):
        self.aerial_ds = aerial_ds
        self.ground_ds = ground_ds

        a_imgs = self.aerial_ds.imgs
        g_imgs = self.ground_ds.imgs

        if len(a_imgs) > len(g_imgs):
            a_imgs = [(p, t) for p, t in a_imgs if (p.replace('aerial', 'ground'), t) in g_imgs]
        if len(g_imgs) > len(a_imgs):
            g_imgs = [(p, t) for p, t in g_imgs if (p.replace('ground', 'aerial'), t) in a_imgs]
        
        self.aerial_ds.imgs = a_imgs
        self.ground_ds.imgs = g_imgs

    def __len__(self):
        return len(self.aerial_ds.imgs)

    def __getitem__(self, index):
        # print('MM', index)
        return self.aerial_ds[index], self.ground_ds[index]

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        # print('IF', index)
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]

def create_dataloader(aerial_data_dir, ground_data_dir, input_size, batch_size, mean_a, std_a, mean_g, std_g):
    data_transforms_a = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_a, std_a)
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean_a, std_a)
    ]),
    }

    data_transforms_g = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean_g, std_g)
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean_g, std_g)
    ]),
    }

    image_datasets = {x: MultiImageFolder(MyImageFolder(os.path.join(aerial_data_dir, x), data_transforms_a[x]), 
                                          MyImageFolder(os.path.join(ground_data_dir, x), data_transforms_g[x])
                                         ) for x in ['train', 'val']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                        batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False) 
                        for x in ['train', 'val']}
    return dataloaders_dict


