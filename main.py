import torch, os
import numpy as np
from datetime import datetime

from DataLoader import TimeSeriesDataset

SleepEEG = TimeSeriesDataset(torch.load(os.path.join("datasets", "SleepEEG", "train.pt")), augment = True, jitter=True, scaling=True)