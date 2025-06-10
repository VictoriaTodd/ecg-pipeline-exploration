import torch
import torch.nn as nn

class ToyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, num_classes=11, dropout=0.2):
        super(ToyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM expects input shape (seq_len, batch, input_size)
        # input_size=1 because input is 1D ECG per timestep
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=False)
        
        # Final classification layer
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        # Add feature dimension: (batch_size, seq_len, input_size=1)
        x = x.unsqueeze(-1)
        
        # LSTM returns output and (hidden, cell)
        # output shape: (batch_size, seq_len, hidden_size)
        output, (hn, cn) = self.lstm(x)
        
        # Use the last hidden state for classification
        # hn shape: (num_layers, batch_size, hidden_size)
        last_hidden = hn[-1]  # Take last layer's hidden state
        
        logits = self.fc(last_hidden)
        return logits
