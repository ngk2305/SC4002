import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, num_classes, word_embedding_size, context_embedding_size,
                 hidden_size_linear):
        super(TextRCNN, self).__init__()
        
        self.text_length = self._length  # No need to pass input_text here

        self.lstm = nn.LSTM(input_size=word_embedding_size, hidden_size=context_embedding_size,
                             bidirectional=True, batch_first=True, dropout=0.5)
        self.linear= nn.Linear(word_embedding_size + 2*context_embedding_size, hidden_size_linear)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size_linear, num_classes)

        self.ConvBlock1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2))
        
        self.ConvBlock2 = nn.Sequential(
            nn.Conv1d(in_channels=96, out_channels=2*context_embedding_size+word_embedding_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),)
        
        self.ConvBlock3 = nn.Sequential(
            nn.Conv1d(in_channels=2*context_embedding_size+word_embedding_size, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=384, out_channels=2*context_embedding_size+word_embedding_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )        
        self.dropout = nn.Dropout(p=0.5)
        
        self.output = nn.Linear(2*context_embedding_size+word_embedding_size, num_classes)
    def forward(self, x):
        # [bs, seq_length, word_embedding_size]

        lstm_out, _ = self.lstm(x)

        # [bs, seq_length, 2*context_embedding_size]
        x = torch.cat([lstm_out, x], 2)

        # [bs, seq_length, word_embedding_size+2*context_embedding_size]
        x = self.linear(x)
        # [bs, seq_length, hidden_size_linear] 
        x = self.tanh(x.permute(0, 2, 1))

        # [bs, hidden_size_linear, seq_length]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # [bs, hidden_size_linear]
        x = x.unsqueeze(-1)
        # [bs, hidden_size_linear, 1]
        x = x.permute(0, 2, 1)
        # [bs, 1, hidden_size_linear]
        x = self.ConvBlock1(x)
        # [bs, conv1 out channel, new size]

        x = self.ConvBlock2(x)
        # [bs, word_embedding_size+2*context_embedding_size, new size]

        x = self.ConvBlock3(x)
        # [bs, word_embedding_size+2*context_embedding_size, new size]

        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
        # [bs, word_embedding_size+2*context_embedding_size]

        x = self.dropout(x)
        
        x = self.output(x)
        # [bs, num_classes]
        return x



    @staticmethod
    def _length(seq):
        relevant = torch.sign(torch.abs(seq))
        length = torch.sum(relevant, dim=1)
        length = length.to(torch.int32)
        return length





