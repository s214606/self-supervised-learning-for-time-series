#%%
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
import torch, os
import numpy as np
from configs.Colors_configs import Colors
colors = Colors()
#%%

valdata = torch.load(os.path.join('datasets/Epilepsy', "test.pt"))
val_label = valdata['labels'].detach().cpu().numpy()

# %%
values, counts = np.unique(val_label, return_counts=True)
pred = np.repeat(values[np.argmax(counts)], len(val_label))
# %%
print(np.mean(pred==val_label))
print(precision_score(val_label,pred, average='weighted'))
print(recall_score(val_label,pred, average='weighted'))
print(f1_score(val_label,pred, average='weighted'))
# %%