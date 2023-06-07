import torch
import numpy as np
from transforms3d.axangles import axangle2mat

def augment_Data_TD(data, do_jitter = False, do_scaling = False, do_rotation = False):
    if do_jitter and do_scaling:
        aug = jitter(scaling(data))
    elif do_jitter:
        aug = jitter(data)
    elif do_scaling:
        aug = scaling(data)
    elif do_rotation:
        aug = rotation(data)
    else:
        return None
    return aug

def augment_Data_FD(data, do_removal = False, do_addition = False):
    if do_removal:
        aug = remove_frequencies(data, aug_ratio=0.8) # "removing 20%" ?? Ask thea about normal distribution
    elif do_addition:
        aug = add_frequencies(data, aug_ratio=0.2)
    else:
        return None
    return aug

def remove_frequencies(data, aug_ratio=0):
    frequencies_to_keep = torch.FloatTensor(data.shape).uniform_() > aug_ratio
    return data*frequencies_to_keep

def add_frequencies(data, aug_ratio=0):
    frequencies_to_add = torch.FloatTensor(data.shape).uniform_() > (1-aug_ratio)
    max_amp = data.max()
    randomized_amps = torch.rand(data.shape)*(max_amp*0.01)
    return data + (frequencies_to_add*randomized_amps)

def jitter(data, sigma = 5):
    # Add noise to every observation within every sample, by sampling from a normal distribution with the same shape as the data
    noise = np.random.normal(loc = 0, scale = sigma, size = data.shape)
    return data + noise

def scaling(data, sigma = 0.1):
    scalingFactor = np.random.normal(loc = 1.0, scale = sigma, size = (data.shape[0], data.shape[2]))
    return np.multiply(data, scalingFactor[:, np.newaxis, :])

def rotation(data):
    axis = np.random.uniform(low=-1, high=1, size=data.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(data , axangle2mat(axis,angle))
