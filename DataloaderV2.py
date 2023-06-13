## Import libraries
import torch, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm



from torch.utils.data import Dataset, DataLoader, random_split
import torch.fft as fft
from Augmentation import augment_Data_FD, augment_Data_TD
from configs.wisdm_configs import Config



class WISDMDataset(Dataset):
    def __init__(self, Xpath, Ypath):
        # #Local config
        self.seed = 0
        np.random.seed(self.seed)
        
        #Load torch data from path
        with open(Xpath, 'rb') as f:
            self.X = torch.load(f)
            
        with open(Ypath, 'rb') as f:
            self.Y = torch.load(f)
            
        self.X_aug = None
        self.X_f_aug = None
        self.y_aug = None
        
        self.X_f = fft.fft(self.X).abs()
        
        if augment: ## Only augment data if we ask for it to be augmented
            self.X_aug = augment_Data_TD(self.X, config, do_jitter = jitter, do_scaling = scaling, do_rotation = rotation)
            self.X_f_aug = augment_Data_FD(self.X_f, do_removal = removal, do_addition = addition)
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx, :, :], self.Y[idx, :, :]
    
    
    
if __name__ == '__main__':
    Xpath = 'datasets\wisdm-dataset_processed\phoneAccel\X_train.pt'
    Ypath = 'datasets\wisdm-dataset_processed\phoneAccel\Y_train.pt'
    dataset = WISDMDataset(Xpath, Ypath)
    print(len(dataset))
    print(dataset[5][1].shape)

    data_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print(next(iter(data_loader))[0].shape)
    