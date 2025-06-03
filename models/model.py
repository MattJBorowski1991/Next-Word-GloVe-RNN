import torch
import torch.nn as nn

class NextWordGloVeRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, embedding_matrix):
        super(NextWordGloVeRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = True  # Fine-tune embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len] -> [batch, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        out = self.fc(lstm_out[:, -1, :])  # Last time step: [batch, vocab_size]
        return out