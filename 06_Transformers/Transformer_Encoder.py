import torch
import torch.nn as nn


class SmallTransformer(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model,
        max_seq_length,
        num_heads,
        num_layers
    ):
        super().__init__()

        # 1. Token Embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        # 2. Position Embedding
        self.position_embedding = nn.Embedding(
            max_seq_length,
            d_model
        )

        # 3. One Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            batch_first=True
        )

        # 4. Stack multiple Encoder Layers
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, token_ids):

        # Get batch size and sequence length
        batch_size, seq_length = token_ids.shape

        # Create position IDs
        positions = torch.arange(
            seq_length,
            device=token_ids.device
        )

        # Convert token IDs into vectors
        token_embeddings = self.token_embedding(
            token_ids
        )

        # Convert positions into vectors
        position_embeddings = self.position_embedding(
            positions
        )

        # Add token information + position information
        x = token_embeddings + position_embeddings

        # Pass through Transformer Encoder
        x = self.encoder(x)

        return x



# Create the Transformer
model = SmallTransformer(
    vocab_size=10000,
    d_model=128,
    max_seq_length=50,
    num_heads=4,
    num_layers=2
)



# Example input

# "I love AI"
# Assume these are token IDs
token_ids = torch.tensor([
    [42, 891, 1532]
])


# Pass input through the Transformer
output = model(token_ids)


# Print results
print("Input shape:")
print(token_ids.shape)

print("\nOutput shape:")
print(output.shape)

print("\nOutput:")
print(output)