import torch
import pandas as pd
from rcnn import TextRCNN
from torch.utils.data import DataLoader, random_split
import ast
from data_helpers import MyDataset
from data_helpers import EarlyStopper
from tqdm import tqdm
from torch import optim
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix

num_classes = 13  # Change to your number of classes
word_embedding_size = 50  # Change to your desired embedding size
context_embedding_size = 50  # Change to your desired context embedding size
cell_type = "lstm"  # Change to "vanilla" or "gru" if needed
loss_fn= torch.nn.CrossEntropyLoss()
batch_size=1


#read/load data
df = pd.read_csv('selected_data.csv')
print('Done read csv file!')
df['word_embeddings']= df['word_embeddings'].apply(ast.literal_eval)
print('Done converting WEm into literals')
df['word_embeddings']= df['word_embeddings'].apply(torch.Tensor)
print('Done converting WEm into tensors')
my_dataset = MyDataset(df)

#train,validation,test split param
validation_ratio = 0.2
test_ratio = 0.2

#Splitting data
validation_size = int(validation_ratio * len(my_dataset))
test_size = int(test_ratio * len(my_dataset))
train_size = len(my_dataset) - validation_size - test_size
train_dataset, validation_dataset, test_dataset = random_split(
    my_dataset, [train_size, validation_size, test_size])

# Create data loaders for training, validation, and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize variables for early stopping
early_stopper = EarlyStopper(patience=3)
model = TextRCNN( num_classes, word_embedding_size, context_embedding_size, cell_type,
                 loss_fn)
optimizer= optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 10
torch.set_grad_enabled(True) 
train_accuracy = []
train_loss = []

test_accuracy = []
test_loss = []

for epoch in tqdm(range(EPOCHS), desc="Training Progress"):
    model.train()

    train_loss_in_epoch = []
    train_accuracy_in_epoch = []

    for batch_x, batch_y in train_loader:
        # forward pass
        # print(batch_x.size())
        optimizer.zero_grad()
        logits  = model(batch_x)
        logits = logits.type(torch.float)
        # print(outputs.size())
        outputs = torch.argmax(logits, dim=-1)

        #print(batch_y)
        one_hot_label = torch.nn.functional.one_hot(batch_y, num_classes=13)
        one_hot_label = one_hot_label.view(-1)
        one_hot_label = one_hot_label.type(torch.float)
        #print(one_hot_label)
        #print(batch_y.size())

        loss = loss_fn(logits, one_hot_label)
        #print(loss)
        loss = torch.autograd.Variable(loss, requires_grad=True)
        train_loss_in_epoch.append(loss)

        acc = (outputs.round() == batch_y).float()
        train_accuracy_in_epoch.append(acc)
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    train_loss.append(torch.mean(torch.stack(train_loss_in_epoch)).item())
    train_accuracy.append(np.mean(train_accuracy_in_epoch))

    model.eval()
    eval_loss_in_epoch = []
    eval_accuracy_in_epoch = []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(
                test_loader,
                desc="Epoch {} Testing".format(epoch+1),
                leave=False):
            # Calculating loss
            dummy = model(batch_x)
            dummy = dummy.type(torch.float)
            predicted_labels= torch.argmax(dummy, dim=-1)

            one_hot_label = torch.nn.functional.one_hot(batch_y, num_classes=13)
            one_hot_label = one_hot_label.view(-1)
            one_hot_label = one_hot_label.type(torch.float)

            loss = loss_fn(dummy, one_hot_label)
            eval_loss_in_epoch.append(loss)

            accuracy = acc = (predicted_labels.round() == batch_y).float().mean()
            eval_accuracy_in_epoch.append(accuracy)

    test_loss.append(torch.mean(torch.stack(eval_loss_in_epoch)).item())
    test_accuracy.append(np.mean(eval_accuracy_in_epoch))

    # Early Stopping
    if early_stopper.early_stop(
            torch.mean(torch.stack(eval_loss_in_epoch)).item()):
        print(f"Early Stop at Epoch {epoch + 1}!")
        break
    print(
    f"Epoch {epoch + 1} | Train Loss {train_loss[-1]:.5f} | Train Acc {train_accuracy[-1]:.5f} | Test Loss {test_loss[-1]:.5f} | Test Acc {test_accuracy[-1]:.5f}"
)

