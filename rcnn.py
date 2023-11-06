import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, sequence_length, num_classes, word_embedding_size, context_embedding_size,
                 cell_type, hidden_size,  loss_function, l2_reg_lambda=0.0,):
        super(TextRCNN, self).__init__()
        self.loss_fn = loss_function
        self.dropout = nn.Dropout()

        self.text_length = self._length  # No need to pass input_text here

        self.bi_rnn = nn.RNN(input_size=word_embedding_size, hidden_size=context_embedding_size,
                             num_layers=1, bidirectional=True, batch_first=True)
        self.linear= nn.Linear(2*context_embedding_size,context_embedding_size)
        self.word_rep = nn.Linear(2 * context_embedding_size + word_embedding_size, hidden_size)
        self.max_pooling = nn.MaxPool1d(sequence_length)
        self.output = nn.Linear(hidden_size, num_classes)

    def forward(self, input_text, labels=None):
        rnn_out, _ = self.bi_rnn(input_text)
        rnn_out = self.linear(rnn_out)
        print(rnn_out.size())
        c_left = torch.cat([torch.zeros_like(rnn_out[:, :1]), rnn_out[:, :-1]], dim=1)
        c_right = torch.cat([rnn_out[:, 1:], torch.zeros_like(rnn_out[:, :1])], dim=1)
        print(c_left.size())
        print(c_right.size())
        x = torch.cat([c_left, input_text, c_right], dim=1)
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
            loss = self.loss_fn
            return loss
        output = torch.argmax(logits, dim=-1)
        return logits, output



    @staticmethod
    def _length(seq):
        relevant = torch.sign(torch.abs(seq))
        length = torch.sum(relevant, dim=1)
        length = length.to(torch.int32)
        return length





