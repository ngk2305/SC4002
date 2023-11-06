import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, sequence_length, num_classes, word_embedding_size, context_embedding_size,
                 cell_type, hidden_size,  loss_fn, l2_reg_lambda=0.0,):
        super(TextRCNN, self).__init__()
        self.loss_fn=loss_fn
        self.dropout = nn.Dropout()

        self.text_length = self._length  # No need to pass input_text here

        self.bi_rnn = nn.RNN(input_size=word_embedding_size, hidden_size=context_embedding_size,
                             num_layers=1, bidirectional=True, batch_first=True)
        self.linear= nn.Linear(2*context_embedding_size,context_embedding_size)
        self.word_rep = nn.Linear(2 * context_embedding_size + word_embedding_size, hidden_size)
        self.max_pooling = nn.MaxPool1d(sequence_length)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, input_text, labels=None):
        input_text = self.dropout(input_text)

        rnn_out, _ = self.bi_rnn(embedded_chars)
        rnn_out = self.linear(rnn_out)
        print(rnn_out.size())
        c_left = torch.cat([torch.zeros_like(rnn_out[:, :1]), rnn_out[:, :-1]], dim=1)
        c_right = torch.cat([rnn_out[:, 1:], torch.zeros_like(rnn_out[:, :1])], dim=1)
        print(c_left.size())
        print(c_right.size())
        x = torch.cat([c_left, embedded_chars, c_right], dim=2)
        print(x.size())
        print(self.word_rep)
        x = self.word_rep(x)
        print(x.size())
        x = x.permute(0, 2, 1)  # Adjust dimensions for max-pooling
        print(x.size())
        x = self.max_pooling(x)
        print(x.size())
        x = x.view(x.size(0), -1)  # Flatten
        print(x.size())
        logits = self.output(x)
        if labels is not None:
            loss = loss_fn
            return loss
        output = torch.argmax(logits, dim=-1)
        return logits, output



    @staticmethod
    def _length(seq):
        relevant = torch.sign(torch.abs(seq))
        length = torch.sum(relevant, dim=1)
        length = length.to(torch.int32)
        return length


# Usage example:
sequence_length = 100  # Change to your desired sequence length
num_classes = 13  # Change to your number of classes
word_embedding_size = 128  # Change to your desired embedding size
context_embedding_size = 128  # Change to your desired context embedding size
cell_type = "lstm"  # Change to "vanilla" or "gru" if needed
hidden_size = 256  # Change to your desired hidden size
loss_fn= torch.nn.CrossEntropyLoss
batch_size=1

model = TextRCNN(sequence_length, num_classes, word_embedding_size, context_embedding_size, cell_type,
                 hidden_size,loss_fn)
input_text = torch.randint(0, vocab_size, (batch_size, sequence_length))  # Change batch_size as needed
logits, output = model(input_text)
print(output)