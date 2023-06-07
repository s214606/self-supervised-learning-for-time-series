## Import libraries
import torch, os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


#load dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, sensorDevice:str, sensor:str):
        self.sensorDevice = sensorDevice #is phone or watch
        self.sensor = sensor #is accel or gyro
        self.basepath = "datasets\\wisdm-dataset\\raw"
        
        self.datapath = os.path.join(self.basepath,self.sensorDevice,self.sensor) #where the data is
        self.filenames = os.listdir(path=self.datapath)
        print(os.path.join(self.datapath,self.filenames[1]))
        
        
if __name__ == '__main__':
    data = TimeSeriesDataset(sensorDevice="phone",sensor="accel")
    
    
        
        
        
        
        








def data_generator(sourcedata_path, targetdata_path, config, 
                   augment = False, jitter = False, scaling = False, permute = False):
    """Load data for pre-training, fine-tuning and for testing."""
    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))
    test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))

    train_dataset = TimeSeriesDataset(train_dataset, config, augment, jitter, scaling, permute)
    finetune_dataset = TimeSeriesDataset(finetune_dataset, config, augment, jitter, scaling, permute)
    test_dataset = TimeSeriesDataset(test_dataset, config, augment, jitter, scaling, permute)

    train_loader = DataLoader(dataset = train_dataset, shuffle = True, batch_size=config.batch_size, drop_last = config.drop_last)
    valid_loader = DataLoader(dataset = finetune_dataset, shuffle = True, batch_size=config.target_batch_size, drop_last = config.drop_last)
    test_loader = DataLoader(dataset = test_dataset, shuffle = True, batch_size=config.target_batch_size, drop_last = config.drop_last)

    return train_loader, valid_loader, test_loader
        