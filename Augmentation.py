import torch
import numpy as np

def augment_Data_TD(data, do_jitter = False, do_scaling = False, do_permute = False):
    if do_jitter and do_scaling:
        aug = jitter(scaling(data))
    elif do_jitter:
        aug = jitter(data)
    elif do_scaling:
        aug = scaling(data)
    elif do_permute:
        aug = permute(data)
    
    return aug

def augment_Data_FD():
    raise(NotImplementedError)

def jitter(data, sigma = 5):
    # Add noise to every observation within every sample, by sampling from a normal distribution with the same shape as the data
    noise = np.random.normal(loc = 0, scale = sigma, size = data.shape)
    return data + noise

def scaling(data, sigma = 0.5):
    # Add the same noise to every observation 
    scalingFactor = np.random.normal(loc = 1.0, scale = sigma, size = (data.shape[0], data.shape[2]))
    return np.multiply(data, scalingFactor[:, np.newaxis, :])

def permute(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

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
