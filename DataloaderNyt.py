## Import libraries
import torch, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.fft as fft
from Augmentation import augment_Data_FD, augment_Data_TD




#load dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, sensorDevice:str, sensor:str, config, augment = False, jitter = False, 
                 scaling = False, rotation = False, removal = False, addition = False):
        # #Local config
        # self.seed = 0
        # np.random.seed(self.seed)
        
        self.sensorDevice = sensorDevice #is phone or watch
        self.sensor = sensor #is accel or gyro
        self.basepath = "datasets\\wisdm-dataset\\raw"
        
        self.datapath = os.path.join(self.basepath,self.sensorDevice,self.sensor) #where the data is
        self.filenames = os.listdir(path=self.datapath)
        #Remove .DS_Store from list if its there
        if '.DS_Store' in self.filenames:
            self.filenames.remove('.DS_Store')
        
        self.NSamples = len(self.filenames)
        
        labelEncoder = LabelEncoder()
        
        self.X = np.empty((self.NSamples,60771,4))
        self.Y = np.empty((self.NSamples,60771))
        self.X_aug = None
        self.X_f_aug = None
        self.y_aug = None
        
        for i in range(self.NSamples):
            path = os.path.join(self.datapath,self.filenames[i])
            print(path)
            df = pd.read_csv(path,
                         header=None, names=["id","label","time","x","y","z"])
            df['z'] = df['z'].str.replace(r';', '')
            
            YSample = np.array(df["label"])
            
            YEncoded = labelEncoder.fit_transform(YSample)
            XSample = df[['time','x', 'y', 'z']].to_numpy()
            
            
            # Sample 60000 time steps from the X data and YEncoded data, where the time steps
            # Are sampled evenly spread out over the time series
            XResampled = np.array([XSample[i] for i in np.linspace(0,len(XSample)-1,60771).astype(int)])
            YResampledEncoded = np.array([YEncoded[i] for i in np.linspace(0,len(YEncoded)-1,60771).astype(int)])
        
            self.X[i,:,:] = XResampled
            self.Y[i,:] = YResampledEncoded

        #X and Y numpy arrays are converted to tensors
        self.X = torch.from_numpy(self.X).float()
        self.Y = torch.from_numpy(self.Y).long()
        
        self.X_f = fft.fft(self.X).abs()
        
        if augment: ## Only augment data if we ask for it to be augmented
            self.X_aug = augment_Data_TD(self.X, config, do_jitter = jitter, do_scaling = scaling, do_rotation = rotation)
            self.X_f_aug = augment_Data_FD(self.X_f, do_removal = removal, do_addition = addition)
        
        
        
        def __len__(self):
            return self.NSamples
        
        
        def __getitem__(self, idx):
            if self.augment:
                return self.X[idx, :, :], self.y[idx, :, :], self.X_aug[idx, :, :],  \
                    self.X_f[idx, :, :], self.X_f_aug[idx, :, :]
            else:
                return self.X[idx, :, :], self.y[idx, :, :], self.X[idx, :, :],  \
                    self.X_f[idx, :, :], self.X_f[idx, :, :]
        
        
        
        

        # first apply label encoding
        
        
        
        
        
        # result = np.empty((self.NSamples,len(X),4))
        
        # print(result.shape)
        # print(result[0,:,:])
        # result[0,:,:] = X
        # print(result[0,:,:])
        
        
        
        
        
        
        
        
        
        
        
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
        