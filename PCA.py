import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from Analysis.load import load_embeddings, load_embeddingsf
import matplotlib.pyplot as plt
import torch
import os
from DataLoader import data_generator
from configs.SleepEEG_configs import Config

config = Config()
sourcedata_path = os.path.join("datasets", "SleepEEG")
targetdata_path = os.path.join("datasets", "Epilepsy")
train_loader, valid_loader, test_loader = data_generator(sourcedata_path, targetdata_path, config)

def Format(z_t, z_f, z_t_aug, z_f_aug):
    # Prepare shape of array
    z_tf = np.concatenate((z_t[0].detach().numpy(), z_f[0].detach().numpy()), axis = 1)
    z_tf_aug = np.concatenate((z_t_aug[0].detach().numpy(), z_f_aug[0].detach().numpy()), axis = 1)
    for i in range(1, len(z_t)):
        # Detach from torch tensor
        t = z_t[i].detach().numpy()
        f = z_f[i].detach().numpy()
        # Concatenate time and frequency embedding along the row axis (1)
        temp = np.concatenate((t, f), axis = 1)
        # Add both embedding along the column axis to the features (axis 0)
        z_tf = np.append(z_tf, temp, axis = 0)
        t = z_t_aug[i].detach().numpy()
        f = z_f_aug[i].detach().numpy()
        temp = np.concatenate((t, f), axis = 1)
        z_tf_aug = np.append(z_tf_aug, temp, axis = 0)
    
    return z_tf, z_tf_aug

def classes(data, z_tf, z_tf_aug = None):
    idx = len(z_tf)
    y = data.dataset.y.detach().numpy()[:int(idx)]
    return y

def doPCA(z_tf, z_tf_aug, y):
    # No augmentation
    from scipy.linalg import svd
    X = z_tf
    N, M = X.shape
    # PCA is then carried out using the PCA algorithm
    Y = X - np.ones((N, 1))*X.mean(0)
    # Standardize the data because of high numbers in one column
    Y = Y*(1/np.std(Y, 0))
    U, S, Vh = svd(Y, full_matrices = False)
    V = Vh.T
    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum()

    threshold = 0.9
    plt.figure()
    plt.plot(range(1,len(rho)+1),rho,'x-')
    plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
    plt.plot([1,len(rho)],[threshold, threshold],'k--')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained')
    plt.legend(['Individual','Cumulative','Threshold'])
    #plt.xticks([1,2,3,4])
    plt.grid()
    plt.show()

    # Plot the projection of the data onto the principal component space
    Z = Y @ V
    # Indices of the principal components to be plotted
    i = 0
    j = 1

    classNames = ["0", "1"]
    ## REMOVE
    Z = Z[2340:]
    C = len(classNames)
    for c in range(C):
        # select indices belonging to class c:
        class_mask = y==c
        plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
    plt.legend(classNames)
    plt.xlabel('PC{0}'.format(i+1))
    plt.ylabel('PC{0}'.format(j+1))
    plt.show()

    pca = PCA(n_components = 2)
    pca.fit_transform(X)
    pca.get_feature_names_out()

#h_ts, z_ts, h_fs, z_fs, h_t_augs, z_t_augs, h_f_augs, z_f_augs = load_embeddings(path = os.path.join("Analysis", "Embedingss"))
h_tsf, z_tsf, h_fsf, z_fsf, h_t_augsf, z_t_augsf, h_f_augsf, z_f_augsf = load_embeddingsf(path = os.path.join("Analysis", "Embeddingss"))

z_tf, z_tf_aug = Format(z_tsf, z_fsf, z_t_augsf, z_f_augsf)
y = classes(valid_loader, z_tf)
doPCA(z_tf, z_tf_aug, y)