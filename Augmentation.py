import torch
import numpy as np

def augment_Data_TD(data, do_jitter = False, do_scaling = False):
    if do_jitter and do_scaling:
        aug = jitter(scaling(data))
    elif do_jitter:
        aug = jitter(data)
    elif do_scaling:
        aug = scaling(data)
    else:
        return data
    return aug

def augment_Data_FD(data, do_removal = False):
    if do_removal:
        aug = remove_frequencies(data, removal_ratio=0.8) # "removing 20%" ?? Ask thea about normal distribution
    else:
        return data
    return aug

def remove_frequencies(data, removal_ratio=0):
    frequencies_to_keep = torch.FloatTensor(data.shape).uniform_() > removal_ratio
    return data*frequencies_to_keep

def jitter(data, sigma = 5):
    # Add noise to every observation within every sample, by sampling from a normal distribution with the same shape as the data
    noise = np.random.normal(loc = 0, scale = sigma, size = data.shape)
    return data + noise

def scaling(data, sigma = 0.1):
    scalingFactor = np.random.normal(loc = 1.0, scale = sigma, size = (data.shape[0], data.shape[2]))
    return np.multiply(data, scalingFactor[:, np.newaxis, :])

