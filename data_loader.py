import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import os
import pickle
import numpy as np
import nibabel as nib
from transforms import random_elastic, add_gaussian_noise


class CovidDataset(Dataset):

    def __init__(
        self, train, fold, 
        data_split_dir = 'Task1_COVID_82_datasplit.pkl',
        image_dir = 'data/COVID-19-CT-Seg_20cases',
        segmentation_dir = 'data/Lung_and_Infection_Mask'
    ):
        
        self.train = train

        with open(data_split_dir, 'rb') as f:
            data_split = pickle.load(f)[fold]['train' if train else 'val']    

        self.x, self.y = [], []

        for fn in data_split:
            self.x.append(
                nib.load(os.path.join(image_dir, fn + '.nii.gz')).get_fdata()
            )
            self.y.append(
                nib.load(os.path.join(segmentation_dir, fn + '.nii.gz')).get_fdata()
            )

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):
        
        x, y = self.x[idx], self.y[idx]

        x = (x - x.mean()) / (x.std() + 1e-8)
        
        if self.train:
            if np.random.rand() < 0.8:
                x = random_elastic(x)
            if np.random.rand() < 0.8:
                x = add_gaussian_noise(x)

        return torch.Tensor(x).unsqueeze(0), torch.LongTensor(y)         # Convert to 1-channel image


def get_train_dataloader(fold, batch_size):
    
    training_set = CovidDataset(train = True, fold = fold)

    train_dataloader = torch.utils.data.DataLoader(training_set, batch_size = batch_size, shuffle = True)
    
    return train_dataloader


def get_test_dataloader(fold, batch_size):

    testing_set = CovidDataset(train = False, fold = fold)

    test_dataloader = torch.utils.data.DataLoader(testing_set, batch_size = batch_size, shuffle = False)

    return test_dataloader
