## Import libraries
import torch, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset
from torch.utils.data import DataLoader



#load dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, sensorDevice:str, sensor:str):
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
        
        Xdata = np.empty((self.NSamples,60771,4))
        Ydata = np.empty((self.NSamples,60771))
        
        for i in range(self.NSamples):
            path = os.path.join(self.datapath,self.filenames[i])
            print(path)
            df = pd.read_csv(path,
                         header=None, names=["id","label","time","x","y","z"])
            df['z'] = df['z'].str.replace(r';', '')
            
            Y = np.array(df["label"])
            
            YEncoded = labelEncoder.fit_transform(Y)
            X = df[['time','x', 'y', 'z']].to_numpy()
            
            
            # Sample 60000 time steps from the X data and YEncoded data, where the time steps
            # Are sampled evenly spread out over the time series
            X = np.array([X[i] for i in np.linspace(0,len(X)-1,60771).astype(int)])
            YEncoded = np.array([YEncoded[i] for i in np.linspace(0,len(YEncoded)-1,60771).astype(int)])
        
            Xdata[i,:,:] = X
            Ydata[i,:] = YEncoded
 
        #Show me how the final Xdata looks like
        print(Xdata.shape)
        print(Xdata[np.random.randint(0,self.NSamples),:,:])
        print(Ydata.shape)
        print(Ydata[np.random.randint(0,self.NSamples),:])
        
        
        

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
        