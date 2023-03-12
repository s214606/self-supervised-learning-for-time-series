## Import libraries
import torch, os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch.fft as fft

class TimeSeriesDataset(Dataset):
    # Initializing this class requires that the parameter 'dataset' is inserted as an already loaded torch tensor using torch.load
    def __init__(self, dataset, augment = False):
        self.X = dataset['samples'].squeeze(1)
        self.y = dataset['labels']

        # Transfer data to frequency domain using torch.fft (frequency fourier transform)
        self.X_f = fft.fft(self.X)#.abs()

        if augment == True: ## Only augment data if we ask for it to be augmented
            raise(NotImplementedError)
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.y)

    def plot_sample(self, dataset):
        # Plot a random sample from the dataset
        plt.plot(dataset[np.random.random_integers(self.__len__())])