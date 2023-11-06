import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.legacy import Field, TabularDataset, BucketIterator, Iterator
import data_helpers

# Parameters
# ==================================================

# Data loading params
pos_dir = "data/rt-polaritydata/rt-polarity.pos"
neg_dir = "data/rt-polaritydata/rt-polarity.neg"

# Evaluation Parameters
batch_size = 64
checkpoint_dir = ""

# Misc Parameters
allow_soft_placement = True
log_device_placement = False

# Load data and labels
x_text, y = data_helpers.load_data_and_labels(pos_dir, neg_dir)

# Map data into vocabulary
text_field = Field(sequential=True, tokenize=lambda x: x.split(), use_vocab=True)
label_field = Field(sequential=False, use_vocab=False, preprocessing=lambda x: int(x))

fields = [('text', text_field), ('label', label_field)]

examples = [torchtext.data.Example.fromlist([x_text[i], y[i]], fields) for i in range(len(x_text))]
dataset = torchtext.data.Dataset(examples, fields)

text_field.build_vocab(dataset)

x_eval = [torchtext.data.example.Example.fromlist([x_text[i]], [('text', text_field)]) for i in range(len(x_text))]

# Define the model
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes, filter_sizes, num_filters, dropout_prob):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = x.permute(1, 0)  # Change batch_size x seq_len to seq_len x batch_size
        embedded = self.embedding(x).unsqueeze(1)
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool2d(conv, (conv.shape[2], 1)).squeeze(3).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, 1)
        cat = self.dropout(cat)
        output = self.fc(cat)
        return output

# Initialize the model
model = TextCNN(len(text_field.vocab), 128, 2, [3, 4, 5], 128, 0.5)

# Load the saved model
checkpoint = torch.load(os.path.join(checkpoint_dir, "model.pth"))
model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
model.eval()

# Prepare the evaluation data
x_eval = [text_field.process([example]) for example in x_eval]
x_eval = torch.stack(x_eval).squeeze(1)

# Generate batches for one epoch
eval_data = torch.utils.data.TensorDataset(x_eval)
eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=batch_size)

# Evaluate the model
correct_predictions = 0
total_examples = 0

with torch.no_grad():
    for inputs in eval_loader:
        inputs = inputs[0].to(device)  # Assuming 'device' is set to 'cuda' or 'cpu'
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total_examples += inputs.size(0)
        correct_predictions += (predicted == y_eval[total_examples - inputs.size(0):total_examples]).sum().item()

print("Total number of test examples: {}".format(total_examples))
print("Accuracy: {:g}".format(correct_predictions / total_examples))