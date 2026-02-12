import torch
import torch.nn as nn

class MAESTRO(nn.Module):
    def __init__(self, input_dim=9, seq_len=750, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 2)

    def forward(self, x):
        # x shape: [Batch, 750, 9]
        x = self.input_fc(x) + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        return self.decoder(x.mean(dim=1))
