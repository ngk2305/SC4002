import torch
import pandas as pd
from rcnn import TextRCNN
import ast

sequence_length = 100  # Change to your desired sequence length
num_classes = 13  # Change to your number of classes
word_embedding_size = 50  # Change to your desired embedding size
context_embedding_size = 50  # Change to your desired context embedding size
cell_type = "lstm"  # Change to "vanilla" or "gru" if needed
hidden_size = 256  # Change to your desired hidden size
loss_fn= torch.nn.CrossEntropyLoss
batch_size=1

model = TextRCNN(sequence_length, num_classes, word_embedding_size, context_embedding_size, cell_type,
                 hidden_size,loss_fn)

df = pd.read_csv('selected_data.csv')
df['word_embeddings'] = df['word_embeddings'].apply(ast.literal_eval)
input_text= torch.Tensor(df.loc[0, 'word_embeddings'])
print(input_text)
input_text.size()


logits, output = model(input_text)
print(output)