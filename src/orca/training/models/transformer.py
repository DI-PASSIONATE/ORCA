import torch.nn as nn
import torch.nn.functional as F
import torch


class SParamTransformer(nn.Module):
    """
    Auto-regressive Transformer model for S-parameter prediction.
    (Will some day) predict S-parameters sequentially given geometry parameters by performing
    self-attention over previously predicted S-parameters.
    """

    def __init__(self, n_inputs=6, s_dim=72, d_model=256, nhead=8, num_layers=6):
        super().__init__()

        # Geometry Embedding
        self.geom_encoder = nn.Sequential(
            nn.Linear(n_inputs, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )

        # S-Parameter Embedding
        self.s_embed = nn.Linear(s_dim, d_model)

        # Frequency Positional Encoding (200 steps + 1 for geom token)
        self.pos_enc = nn.Parameter(torch.zeros(1, 201, d_model))

        # Transformer Encoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Output Layer
        self.fc_out = nn.Linear(d_model, s_dim)

    def forward(self, geom, s_prev):
        """
        geom:   (Batch, n_inputs) -> e.g., (32, 6)
        s_prev: (Batch, seq_len, 72) -> The sequence of S-params seen so far
        """
        # Embed the geometry to act as the sequence starter
        geom_token = self.geom_encoder(geom).unsqueeze(1)  # (B, 1, d_model)

        # Embed the S-parameter sequence
        s_tokens = self.s_embed(s_prev)  # (B, seq_len, d_model)

        # Concat: [Geom, S_1, S_2, ... S_n]
        x = torch.cat([geom_token, s_tokens], dim=1)

        # Add positions & Apply Causal Mask
        x = x + self.pos_enc[:, : x.size(1), :]
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)

        out = self.transformer(x, mask=mask)

        # We only want to predict the S-parameters, so skip the first output (geom)
        return self.fc_out(out[:, :, :])
