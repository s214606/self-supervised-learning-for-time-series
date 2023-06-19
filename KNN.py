import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import torch, os
from configs.Colors_configs import Colors
colors = Colors()

traindata = torch.load(os.path.join('datasets/Epilepsy', "test.pt"))
valdata = torch.load(os.path.join('datasets/Epilepsy', "val.pt"))

train_fea, train_label = (
    traindata['samples'].detach().cpu().numpy().squeeze(1),
    traindata['labels'].detach().cpu().numpy()
)
val_fea, val_label = (
    valdata['samples'].detach().cpu().numpy().squeeze(1),
    valdata['labels'].detach().cpu().numpy()
)

n_neighbors_values = range(1, 21)
accuracy_values = []

for n_neighbors in n_neighbors_values:
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(train_fea, train_label)

    val_acc = neigh.score(val_fea, val_label)
    accuracy_values.append(val_acc)

plt.plot(n_neighbors_values, accuracy_values, marker='o', color=colors.secondary, markerfacecolor=colors.primary, markersize = 8)
plt.xlabel('N neighbors')
plt.ylabel('Accuracy')
plt.title('KNN accuracy with varying N neighbours')
plt.show()

# Metrics
from sklearn.metrics import precision_score, recall_score, f1_score
neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(train_fea, train_label)
print(neigh.score(val_fea, val_label))
pred = neigh.predict(val_fea)
print("Precision score:", precision_score(val_label,pred, average='weighted'))
print("Recall score:", recall_score(val_label,pred, average='weighted'))
print("F1 score:", f1_score(val_label,pred, average='weighted'))
