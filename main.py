from BotRGCN.model import BotRGCN
from BotRGCN.Dataset import Twibot20
import torch
from torch import nn
from BotRGCN.utils import accuracy,init_weights
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding_size,dropout,lr,weight_decay=128,0.3,1e-3,5e-3

dataset=Twibot20(device=device,process=True,save=True)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,train_idx,val_idx,test_idx=dataset.dataloader()
print("dataloader initialised")

model=BotRGCN(num_prop_size=5,cat_prop_size=3,embedding_dimension=embedding_size).to(device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)
print("model initialised")

train_loss = []
val_loss = []
train_acc = []
val_acc = []

def train(epoch):
    model.train()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output[train_idx], labels[train_idx])
    acc_train = accuracy(output[train_idx], labels[train_idx])
    acc_val = accuracy(output[val_idx], labels[val_idx])
    loss_val = loss(output[val_idx], labels[val_idx])
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'loss_val: {:.4f}'.format(loss_val.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        'acc_val: {:.4f}'.format(acc_val.item()),)
    train_loss.append(loss_train.item())
    val_loss.append(loss_val.item())
    train_acc.append(acc_train.item())
    val_acc.append(acc_val.item())
    return acc_train,loss_train

def test():
    model.eval()
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_test = loss(output[test_idx], labels[test_idx])
    acc_test = accuracy(output[test_idx], labels[test_idx])
    output=output.max(1)[1].to('cpu').detach().numpy()
    label=labels.to('cpu').detach().numpy()

    # Replace 'output', 'test_idx', and 'labels' with your actual variables
    predicted_classes = output[test_idx]
    true_classes = labels[test_idx]

    # Compute the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)

    f1=f1_score(label[test_idx],output[test_idx])
    mcc=matthews_corrcoef(label[test_idx], output[test_idx])
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "f1_score= {:.4f}".format(f1.item()),
            "mcc= {:.4f}".format(mcc.item()),
            )
    return acc_test,loss_test,f1_score,cm

model.apply(init_weights)

epochs=100
metrics = []
for epoch in range(epochs):
    metrics.append(train(epoch))
    
d1,d2,d3,cm = test()

# Plotting train_acc and val_acc
plt.plot(train_acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
# plt.show()

# Save the plot
plt.savefig('mod_training_validation_accuracy.png')

# Save the data as numpy arrays
np.save('Data/train_acc.npy', train_acc)
np.save('Data/val_acc.npy', val_acc)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# plt.show()

# Save the plot
plt.savefig('training_val_loss.png')

# Save the data as numpy arrays
np.save('Data/train_loss.npy', train_loss)
np.save('Data/val_loss.npy', val_loss)

# Save the figure
plt.savefig('confusion_matrix.png')

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
# plt.show()

# Save the figure
plt.savefig('confusion_matrix.png')
