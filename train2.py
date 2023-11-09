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
import os

current_directory = os.getcwd()
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

model = TextRCNN( num_classes, word_embedding_size, context_embedding_size, cell_type,
                 loss_fn)
try:
    model.load_state_dict(torch.load("model.pth"))
except:
    pass

optimizer= optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 100

for epoch in range(EPOCHS):
    model.train()  # Set the model in training mode
    running_loss = 0.0
    for inputs, labels in train_loader:  # Assuming you have a DataLoader for the training dataset


        optimizer.zero_grad()  # Zero the gradient buffers
        outputs = model(inputs)  # Forward pass

        one_hot_label = torch.nn.functional.one_hot(labels, num_classes=13)
        one_hot_label = one_hot_label.view(-1)
        one_hot_label = one_hot_label.type(torch.float)

        loss = loss_fn(outputs, one_hot_label)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {running_loss / len(train_loader)}")

torch.save(model.state_dict(), "model.pth")