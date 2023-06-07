import os
import numpy as np
import torch
from configs.SleepEEG_configs import Config
from DataLoader import data_generator
import matplotlib.pyplot as plt

path = os.path.join("datasets", "SleepEEG")
config = Config()
train_loader, valid_loader, test_loader = data_generator(path, path, config, augment = True, jitter = True, scaling = True)
plt.plot()