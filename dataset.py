import torch
from torch.utils.data import Dataset


class CustomTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        # Implement code here to convert a row of your CSV into a data sample
        # For example, you can return a tuple (input, target)
        input_data = sample['word_embeddings']
        target = sample['numerical_label']
        return input_data, target


def collate_fn(data, args, pad_idx=0):
    """Padding"""
    texts, labels = zip(*data)
    texts = [s + [pad_idx] * (args.max_len - len(s)) if len(s) < args.max_len else s[:args.max_len] for s in texts]
    return torch.LongTensor(texts), torch.LongTensor(labels)

def load_glove_vectors(glove_file):
    word_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            vector = [float(val) for val in values[1:]]
            word_vectors[word] = vector
    return word_vectors

def get_word_embeddings(text,glove_vectors):
    embeddings = []
    for word in text.split():
        if word in glove_vectors:
            embeddings.append(glove_vectors[word])
    return embeddings

