import torch
import numpy as np
from transforms3d.axangles import axangle2mat

def augment_Data_TD(data, configs, do_jitter = False, do_scaling = False, do_permute = False, do_rotation = False):
    if do_jitter and do_scaling:
        aug = jitter(scaling(data, configs), configs)
    elif do_jitter:
        aug = jitter(data, configs)
    elif do_scaling:
        aug = scaling(data, configs)
    elif do_permute:
        aug = permute(data)
    elif do_rotation:
        aug = rotation(data)
    else:
        return data
    return aug

def augment_Data_FD(data, do_removal = False, do_addition = False):
    if do_removal:
        aug = remove_frequencies(data, aug_ratio=0.8) # "removing 20%" ?? Ask thea about normal distribution
    elif do_addition:
        aug = add_frequencies(data, aug_ratio=0.2)
    else:
        return data
    return aug

def remove_frequencies(data, aug_ratio=0):
    frequencies_to_keep = torch.FloatTensor(data.shape).uniform_() > aug_ratio
    return data*frequencies_to_keep

def add_frequencies(data, aug_ratio=0):
    frequencies_to_add = torch.FloatTensor(data.shape).uniform_() > (1-aug_ratio)
    max_amp = data.max()
    randomized_amps = torch.rand(data.shape)*(max_amp*0.01)
    return data + (frequencies_to_add*randomized_amps)

def jitter(data, configs, sigma = 5):
    # Add noise to every observation within every sample, by sampling from a normal distribution with the same shape as the data
    noise = np.random.normal(loc = 0, scale = configs.augmentation.jitter_ratio, size = data.shape)
    return data + noise

def scaling(data, configs, sigma = 0.5):
    # Add the same noise to every observation 
    scalingFactor = np.random.normal(loc = 1.0, scale = configs.augmentation.jitter_scale_ratio, size = (data.shape[0], data.shape[2]))
    return np.multiply(data, scalingFactor[:, np.newaxis, :])

def permute(x, configs, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, configs.augmentation.max_seq, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def rotation(data):
    axis = np.random.uniform(low=-1, high=1, size=data.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(data , axangle2mat(axis,angle))

"""def permute(data, NSeg = 5):
    import time
    start = time.time()
    split_idx = torch.arange(0, data.shape[2], np.round(data.shape[2] / NSeg), dtype = torch.long)
    for i in range(len(data)):
        x = data[i]
        x = torch.tensor_split(x, tensor_indices_or_sections = split_idx[1:], dim = 1)
        numbers = np.arange(5, dtype = int)
        np.random.shuffle(numbers)
        permute_idx = torch.as_tensor(numbers)
        temp = x
        x = torch.zeros()
        for i in range(len(permute_idx)):
            x[i][0] = temp[permute_idx[i]][0]
        x = torch.cat(x, dim = 0)
        data[i] = x

    permuted = x
    end = time.time()
    print("Time elapsed: ", end - start)
    return permuted"""

"""def permute(data, N = 5, minLen = 10):
    # Split data into N slices and randomly change their temporal location
    permuted = np.zeros(data.shape)
    idx = np.random.permutation(N)

    segmentLengthOk = False
    while segmentLengthOk == False:
        segments = np.zeros(N + 1, dtype = int)
        segments[1:-1] = np.sort(np.random.randint(minLen, data.shape[0] - minLen, N - 1))
        segments[-1] = data.shape[0]

        if np.min(segments[1:] - segments[0:-1]) > minLen:
            segmentLengthOk = True
    
    pp = 0
    for i in range(N):
        data_temp = data[segments[idx[i]]:segments[idx[i] + 1], :]
        permuted[pp:pp + len(data_temp), :] = data_temp
        pp += len(data_temp)
    return permuted"""

#def magnitudeWarp(data):
