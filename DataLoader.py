## Import libraries
import torch, os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch.fft as fft
from Augmentation import augment_Data_FD, augment_Data_TD

## Commit#: Find commits with conv1D as model

# Seed for reproducability
seed = 32
np.random.seed(seed)
torch.manual_seed(seed)

class TimeSeriesDataset(Dataset):
    """Load and prepare a dataset for training on a neural network
    Initializing this class requires that the parameter 'dataset' is inserted as an already loaded torch tensor using torch.load.
    """
    def __init__(self, dataset,
                 augment = False,
                 jitter = False, scaling = False, permute = False):
        self.X = dataset['samples']
        self.y = dataset['labels']
        self.X_aug = None
        self.y_aug = None

        # Shuffle data
        data = list(zip(self.X, self.y))
        np.random.shuffle(data)
        self.X, self.y = zip(*data)
        self.X, self.y = torch.stack(list(self.X), dim=0), torch.stack(list(self.y), dim=0)

        if len(self.X.shape) < 3:
            self.X = self.X.unsqueeze(2)

        # Make sure the channel is in the second dimension
        if self.X.shape.index(min(self.X.shape)) != 1:
            self.X = self.X.permute(0, 2, 1)

                
        # Transfer data to frequency domain using torch.fft (frequency fourier transform)
        self.X_f = fft.fft(self.X).abs()
        
        if augment: ## Only augment data if we ask for it to be augmented
            self.X_aug = augment_Data_TD(self.X, do_jitter = jitter, do_scaling = scaling, do_permute = permute)
            #self.X_aug_f = augment_Data_FD(self.X_f)
    
    def __len__(self):
        # Return the length of the dataset
        return len(self.X.shape[0])

    def plot_sample(self, TxF = False, OrigxAug = False):
        # For iterating over multiple plots: plt.figure, then, plt.add_subplot
        plot_idx = np.random.randint(self.__len__())
        # Plot a random sample from the dataset in time- and frequency domain
        if TxF:
            fig, ax = plt.subplots(1,2)
            ax[0].plot(self.X[plot_idx][0])
            ax[0].set_title("Time domain")
            ax[1].plot(self.X_f[plot_idx][0])
            ax[1].set_title("Frequency domain")
            plt.show()
        elif OrigxAug:
            plt.figure(figsize=(12,4))
            plt.plot(self.X[plot_idx][0], label = "Original")
            plt.plot(self.X_aug[plot_idx][0], label = "Augmented")
            plt.legend()
            plt.show()
    
def data_generator(sourcedata_path, targetdata_path,
                   augment = False, jitter = False, scaling = False, permute = False):
    """Load data for pre-training, fine-tuning and for testing."""
    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))
    test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))

    train_dataset = TimeSeriesDataset(train_dataset, augment, jitter, scaling, permute)
    finetune_dataset = TimeSeriesDataset(finetune_dataset, augment, jitter, scaling, permute)
    test_dataset = TimeSeriesDataset(test_dataset, augment, jitter, scaling, permute)