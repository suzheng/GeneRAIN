import torch
import torch.nn as nn
import torch.optim as optim

# Assuming token_emb is your 2D tensor with shape [num_samples, embedding_dim]
# and that you've already loaded it.

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, latent_dim)
        self.mu = nn.Linear(latent_dim, latent_dim)
        self.log_var = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_dim, output_dim)

    def forward(self, z):
        return torch.sigmoid(self.linear(z))

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var