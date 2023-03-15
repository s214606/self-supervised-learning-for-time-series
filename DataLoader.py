## Import libraries
import torch, os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch.fft as fft

# Seed for reproducability
np.random.seed(32)

class TimeSeriesDataset(Dataset):
    # Initializing this class requires that the parameter 'dataset' is inserted as an already loaded torch tensor using torch.load
    def __init__(self, dataset,
                 augment = False,
                 jitter = False, phase = False):
        self.X = dataset['samples']
        self.y = dataset['labels']

        # Make sure the channel is in the second dimension
        if SleepEEG.X.shape.index(min(SleepEEG.X.shape)) != 1:
            self.X = self.X.permute(0, 2, 1)

        if augment: ## Only augment data if we ask for it to be augmented
            raise(NotImplementedError)
                
        # Transfer data to frequency domain using torch.fft (frequency fourier transform)
        self.X_f = fft.fft(self.X)#.abs()
        #self.X_aug_f = fft.fft(self.X_aug)
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.y)

    def plot_sample(self):
        # Plot a random sample from the dataset in time- and frequency domain
        fig, ax = plt.subplots(1,2)
        ax[0].plot(self.X[np.random.randint(self.__len__())])
        ax[0].set_title("Time domain")
        ax[1].plot(self.X_f[np.random.randint(self.__len__())])
        ax[1].set_title("Frequency domain")
        plt.show()

SleepEEG = TimeSeriesDataset(dataset=torch.load(os.path.join("datasets", "SleepEEG", "train.pt")))
#SleepEEG.plot_sample()
print(len(SleepEEG.X.shape))
print(SleepEEG.X.shape.index(min(SleepEEG.X.shape)))