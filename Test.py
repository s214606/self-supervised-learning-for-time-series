## Import libraries
import torch, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

import torch.fft as fft

path1 = "datasets\\wisdm-dataset_processed\\trainIdx.pt"
path2 = "datasets\\wisdm-dataset_processed\\testIdx.pt"
path3 = "datasets\\wisdm-dataset_processed\\valIdx.pt"
#open path1 and path2 and path3 and show the data in them 
with open(path1, 'rb') as f:
    train_idx = torch.load(f)
    print("train person index:")
    print(train_idx)
    
with open(path2, 'rb') as f:
    test_idx = torch.load(f)
    print("test person index:")
    print(test_idx)
    
with open(path3, 'rb') as f:
    val_idx = torch.load(f)
    print("val person index:")
    print(val_idx)
