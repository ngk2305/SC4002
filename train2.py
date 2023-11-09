import torch
import pandas as pd
from rcnn import TextRCNN
from torch.utils.data import DataLoader, random_split
import ast
import torch.nn.functional as F
from data_helpers import MyDataset, EarlyStopper, metrics
from tqdm import tqdm
from torch import optim
import numpy as np
import os

current_directory = os.getcwd()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

num_classes = 13  # Change to your number of classes
word_embedding_size = 50  # Change to your desired embedding size
context_embedding_size = 50  # Change to your desired context embedding size
hidden_size_linear = 300 # change to your hidden layer size
cell_type = "lstm"  # Change to "vanilla" or "gru" if needed
loss_fn = torch.nn.CrossEntropyLoss()
batch_size = 1

model = TextRCNN(num_classes, word_embedding_size, context_embedding_size, hidden_size_linear=hidden_size_linear).to(device)

#read/load data
path = os.getcwd()
df = pd.read_csv(path+'/selected_data.csv')
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

model = TextRCNN(num_classes, word_embedding_size, context_embedding_size, hidden_size_linear=hidden_size_linear).to(device)
try:
    model.load_state_dict(torch.load("model.pth"))
except:
    pass

optimizer= optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 100

losses_ = []
accuracies_ = []
precisions_ = []
recall_ = []
f1_ = []

# metric for best model
best = 1000

for epoch in tqdm(range(EPOCHS), desc='Training Progress'):
    model.train()  # Set the model in training mode
    running_loss = 0.0
    correct = 0
    predictions_, labels_ = [], []
    
    for step, (inputs, labels) in enumerate(train_loader):  # Assuming you have a DataLoader for the training dataset
        
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        running_loss += loss.item() 

        
        optimizer.zero_grad()  # Zero the gradient buffers
        loss.backward() # backpropagation
        optimizer.step() #update parameters
        
        prediction = torch.max(outputs, 1)[1]
        predictions_ += prediction.tolist()
        labels_ += labels.tolist()
        correct += (prediction==labels).sum().item()
        
    avg_loss, accuracy, precision, recall, f1, _ = metrics(train_loader, running_loss, correct, predictions_, labels_)
    losses_.append(avg_loss)
    accuracies_.append(accuracy)
    precisions_.append(precision)
    recall_.append(recall)
    f1_.append(f1)
    if avg_loss < best:
        best = avg_loss
        torch.save(model.state_dict(), 'best.pth')
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss}, Accuracy: {accuracy}, f1: {f1}")
    
torch.save(model.state_dict(), "last.pth")