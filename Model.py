import torch
import numpy as np
import os

from torch import nn

from DataLoader import TimeSeriesDataset

class TFC_Classifer(nn.Module):
    def __init__(self):
        super(TFC_Classifer, self).__init__()
        self.model = 