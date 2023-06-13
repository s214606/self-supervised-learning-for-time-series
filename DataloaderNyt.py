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




#load dataset
class WISDMDataset(Dataset):
    def __init__(self, Xpath, Ypath, config = None, augment = False, jitter = False, 
                 scaling = False, rotation = False, removal = False, addition = False):
        # #Local config
        self.seed = 0
        np.random.seed(self.seed)
        self.augment = augment
        self.jitter = jitter
        self.scaling = scaling
        self.rotation = rotation
        self.removal = removal
        self.addition = addition
        
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
        if self.augment:
            return self.X[idx, :, :], self.Y[idx, :, :], self.X_aug[idx, :, :],  \
                self.X_f[idx, :, :], self.X_f_aug[idx, :, :]
        else:
            return self.X[idx, :, :], self.Y[idx, :, :], self.X[idx, :, :],  \
                self.X_f[idx, :, :], self.X_f[idx, :, :]
        
def data_generator(sourcedata_path_X, sourcedata_path_Y, targetdata_path_X, targetdata_path_Y, config, 
                   augment = False, jitter = False, scaling = False, rotation = False,
                   removal = False, addition = False):

    
    train_dataset = WISDMDataset(sourcedata_path_X, sourcedata_path_Y, config, augment, jitter, scaling, rotation, removal, addition)
    finetune_dataset = WISDMDataset(targetdata_path_X, targetdata_path_Y, config, augment, jitter, scaling, rotation, removal, addition)
    test_dataset = WISDMDataset(targetdata_path_X, targetdata_path_Y, config, augment, jitter, scaling, rotation, removal, addition)

    train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size=config.batch_size, drop_last = config.drop_last)
    valid_loader = DataLoader(dataset = finetune_dataset, shuffle = True, batch_size=config.target_batch_size, drop_last = config.drop_last)
    test_loader = DataLoader(dataset = test_dataset, shuffle = True, batch_size=config.target_batch_size, drop_last = config.drop_last)

    return train_loader, valid_loader, test_loader
        
   
        
if __name__ == '__main__':
    config = Config()
    
    dataset = WISDMDataset("datasets\\wisdm-dataset_processed\\phoneAccel\\X_train.pt", "datasets\\wisdm-dataset_processed\\phoneAccel\\Y_train.pt")
    print(dataset)
    print(dataset.X)
    print(dataset.Y)
    print(len(dataset))
    print(dataset.__len__())
    print(dataset.__getitem__(0))
    print(dataset[0])
    # train_loader, valid_loader, test_loader = data_generator(sourcedata_path_X = "datasets\wisdm-dataset_processed\phoneAccel\X_train.pt",sourcedata_path_Y="datasets\wisdm-dataset_processed\phoneAccel\Y_train.pt",
    #                                                          targetdata_path_X="datasets\wisdm-dataset_processed\phoneAccel\X_Val.pt",targetdata_path_Y="datasets\wisdm-dataset_processed\phoneAccel\Y_Val.pt",
    #                                                          config = config, augment = None, jitter = None, scaling = None, rotation = None, removal = None, addition = None)
    # print(next(iter(train_loader)))
    # print(next(iter(valid_loader)))
    # print(next(iter(test_loader)))
    # print(len(train_loader))
    # print(len(valid_loader))
    # print(len(test_loader))
    
        
        
        
        
        





