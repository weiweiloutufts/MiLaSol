import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinCNN(nn.Module):
    def __init__(self, embed_dim, num_filters, kernel_size, dropout_rate=0.3):
        super(ProteinCNN, self).__init__()

        # 1D CNN layer for feature extraction
        self.cnn = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Batch normalization for stable training
        self.batch_norm = nn.BatchNorm1d(num_filters)

        # Fully connected layer for embedding transformation
        self.fc = nn.Linear(
            num_filters, num_filters
        )  # Keep num_filters if it's used in BiLSTM

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        # print("CNN input:", x.shape)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        # print("CNN input permute:", x.shape)
        x = self.cnn(x)  # (batch, num_filters, seq_len)
        # print("CNN output:", x.shape)
        x = self.batch_norm(F.gelu(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # (batch, seq_len, num_filters)
        # print("CNN output permute:", x.shape)
        return x


class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.GELU(),
            nn.Linear(4096, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


class ProteinClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_filters,
        kernel_size,
        lstm_hidden_dim,
        num_lstm_layers,
        esm_dim,
        prot_dim,
        output_dim,
        latent_dim=64,
    ):
        super(ProteinClassifier, self).__init__()

        # Embedding layer for amino acid sequences
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0
        )

        # CNN for raygun-based feature extraction
        self.cnn = ProteinCNN(embed_dim, num_filters, kernel_size)
        self.cnnbn = nn.BatchNorm1d(num_filters)

        # BiLSTM for sequential modeling on CNN output
        self.lstm = nn.LSTM(
            input_size=num_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Self-attention mechanism
        self.self_attn = nn.MultiheadAttention(
            embed_dim=lstm_hidden_dim * 2, num_heads=4, batch_first=True
        )

        self.bn = nn.LayerNorm(lstm_hidden_dim * 2)
        self.dropout = nn.Dropout(0.3)

        self.pool_proj = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, 1280),
            nn.GELU(),
            nn.LayerNorm(1280),
            nn.Dropout(0.3),
        )

        self.attn2 = nn.MultiheadAttention(
            embed_dim=1280, num_heads=8, batch_first=True
        )
        self.bn2 = nn.LayerNorm(1280)

        self.prot_proj = nn.Sequential(
            nn.Linear(prot_dim, 1280),
            nn.GELU(),
            nn.LayerNorm(1280),
            nn.Dropout(0.2),
        )
        self.attn_prot = nn.MultiheadAttention(
            embed_dim=1280, num_heads=8, batch_first=True
        )
        self.bn_prot = nn.LayerNorm(1280)

        total_dim = 7 * 1280 + lstm_hidden_dim * 2

        self.dae = DenoisingAutoencoder(input_dim=total_dim, hidden_dim=latent_dim)

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.Tanh(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim // 2, output_dim),
        )

    def forward(
        self, sequence_input, esm_input, prot_input, raygun_input, solu_feats=None
    ):

        embedded_input = self.embedding(sequence_input)
        # Apply CNN on sequence input
        cnn_emb = self.cnn(embedded_input)

        # Apply BiLSTM on CNN features
        lstm_out, _ = self.lstm(cnn_emb)  # [B, L, H]
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        attn_out = self.bn(attn_out + lstm_out)

        pooled = attn_out.mean(dim=1)  # [batch_size, hidden_dim * 2]

        query = raygun_input.unsqueeze(1)
        keyval = esm_input.unsqueeze(1)  # [B, 1, 1280]
        attn2_output, _ = self.attn2(query, keyval, keyval)
        attn2_output = self.dropout(attn2_output)
        attn2_output = self.bn2(attn2_output)
        attn2_output = attn2_output.squeeze(1)  # [1, 1280]

        prot_emb = self.prot_proj(prot_input)
        keyval_prot = prot_emb.unsqueeze(1)
        attn_prot, _ = self.attn_prot(query, keyval_prot, keyval_prot)
        attn_prot = self.dropout(attn_prot)
        attn_prot = self.bn_prot(attn_prot)
        attn_prot = attn_prot.squeeze(1)

        interaction_add = esm_input + raygun_input

        interaction = esm_input * raygun_input

        pooled_proj = self.pool_proj(pooled)

        interaction_pool = pooled_proj * esm_input

        fused_inputs = [
            pooled,
            esm_input,
            raygun_input,
            interaction,
            interaction_add,
            interaction_pool,
            attn2_output,
            attn_prot,
        ]
        fused = torch.cat(fused_inputs, dim=1)

        # Add noise to input
        fused = self.dropout(fused)

        noise = torch.randn_like(fused) * 0.1
        fused_noisy = fused + noise

        # Noisy path (for DAE training)
        denoised_fused, z = self.dae(fused_noisy)
        reconstruction_loss = F.mse_loss(denoised_fused, fused)

        # lam = torch.rand(z.size(0), 1, device=z.device)  # shape: [batch_size, 1]
        lam = torch.empty(z.size(0), 1, device=z.device).uniform_(0.75, 0.95)
        index = torch.randperm(z.size(0)).to(z.device)  # shuffled indices

        # Create augmented embedding via interpolation (safe)
        aug_z = lam * z + (1 - lam) * z[index]

        output = self.fc(aug_z)
        return output, z, aug_z, reconstruction_loss
