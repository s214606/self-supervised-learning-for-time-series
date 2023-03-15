import torch
import numpy as np

def augment_Data_TD():
    raise(NotImplementedError)

def augment_Data_FD():
    raise(NotImplementedError)

def augment_Data(self, XData, yData, subset_range = None,
             jitter = False, jitter_range = None,
             phase = False, phase_amount = None):
        # Augment data to add noise to it for training a model invariant for noise
        # Add  noise to every observation within each sample
        if jitter:
            if subset_range is not None:
                for i in subset_range:
                    for j in range(1):
                        pass
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