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




#load dataset
class WISDMDataset(Dataset):
    def __init__(self, Xpath, Ypath, config = None, augment = False, jitter = False, 
                 scaling = False, rotation = False, removal = False, addition = False):
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
        #The train data set is created
        
        self.X_f = fft.fft(self.X).abs()
        
        if augment: ## Only augment data if we ask for it to be augmented
            self.X_aug = augment_Data_TD(self.X, config, do_jitter = jitter, do_scaling = scaling, do_rotation = rotation)
            self.X_f_aug = augment_Data_FD(self.X_f, do_removal = removal, do_addition = addition)
        
        
        def __len__(self):
            return self.X.shape[0]
        
        
        def __getitem__(self, idx):
            if self.augment:
                return self.X[idx, :, :], self.y[idx, :, :], self.X_aug[idx, :, :],  \
                    self.X_f[idx, :, :], self.X_f_aug[idx, :, :]
            else:
                return self.X[idx, :, :], self.y[idx, :, :], self.X[idx, :, :],  \
                    self.X_f[idx, :, :], self.X_f[idx, :, :]
        
def data_generator(sourcedata_path_X, sourcedata_path_Y, targetdata_path_X, targetdata_path_Y, config, 
                   augment = False, jitter = False, scaling = False, permute = False, rotation = False,
                   removal = False, addition = False):

    
    train_dataset = WISDMDataset(sourcedata_path_X, sourcedata_path_Y, config, augment, jitter, scaling, permute, rotation, removal, addition)
    finetune_dataset = WISDMDataset(targetdata_path_X, targetdata_path_Y, config, augment, jitter, scaling, permute, rotation, removal, addition)
    test_dataset = WISDMDataset(targetdata_path_X, targetdata_path_Y, config, augment, jitter, scaling, permute, rotation, removal, addition)

    train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size=config.batch_size, drop_last = config.drop_last)
    valid_loader = DataLoader(dataset = finetune_dataset, shuffle = True, batch_size=config.target_batch_size, drop_last = config.drop_last)
    test_loader = DataLoader(dataset = test_dataset, shuffle = True, batch_size=config.target_batch_size, drop_last = config.drop_last)

    return train_loader, valid_loader, test_loader
        
   
        
if __name__ == '__main__':
    pass    
    
        
        
        
        
        





