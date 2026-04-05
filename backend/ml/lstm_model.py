import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=6, lstm_units=64, hidden_dim=32, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        self.fc1 = nn.Linear(lstm_units, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last hidden state
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return torch.softmax(out, dim=1)  # Return probabilities