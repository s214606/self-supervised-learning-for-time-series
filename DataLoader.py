## Import libraries
import torch, os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch.fft as fft

# Seed for reproducability
np.random.seed(32)

def augment_Data(XData, yData,
                 jitter = False, jitter_amount = None,
                 phase = False, phase_amount = None):
    # Add random noise to every observation within each sample
    if jitter:
        for j in range(len(yData)):
            for i in range(len(yData)):
                XData[j][i] += np.random.rand(jitter_amount)
    # Add noise to every observation in a sample
    if phase:
        for i in range(len(yData)):
            XData[i] += phase_amount
    
    return XData


class TimeSeriesDataset(Dataset):
    # Initializing this class requires that the parameter 'dataset' is inserted as an already loaded torch tensor using torch.load
    def __init__(self, dataset,
                 augment = False,
                 jitter = False, phase = False):
        self.X = dataset['samples'].squeeze(1)
        self.y = dataset['labels']

        if augment: ## Only augment data if we ask for it to be augmented
            raise(NotImplementedError)
                
        # Transfer data to frequency domain using torch.fft (frequency fourier transform)
        self.X_f = fft.fft(self.X)#.abs()
        self.X_aug_f = fft.fft(self.X_aug)
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.y)

    def plot_sample(self, dataset):
        # Plot a random sample from the dataset
        plt.plot(dataset[np.random.random_integers(self.__len__())])

    def augment_Data(self, XData, yData, subset_range = None,
             jitter = False, jitter_range = None,
             phase = False, phase_amount = None):
        # Augment data to add noise to it for training a model invariant for noise
            
        # Add  noise to every observation within each sample
        if jitter:
            if subset_range is not None:
                for i in range(subset_range):
                    for j in range(1)
            else:
                for j in range(self.__len()):
                    for i in range(len(yData)):
                        XData[j][i] += np.random.rand(jitter_range)
            
        # Add noise to every observation in a sample
        if phase:
            for i in range(len(yData)):
                if subset:
                    pass
                else:
                    XData[i] += phase_amount
    
        return XData