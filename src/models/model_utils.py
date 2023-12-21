import torch
import torch.nn as nn
import torch.optim as optim
from models.VAE2L import VAE
import hashlib
from sklearn.model_selection import train_test_split

def vae_loss(reconstructed_x, x, mu, log_var):
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(reconstructed_x, x, reduction='mean')
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss

def compress_emb(token_emb):
    # Convert the data to a NumPy array if it's a tensor
    if isinstance(token_emb, torch.Tensor):
        token_emb = token_emb.detach().cpu().numpy()
    
    # Split the data into training and validation sets
    train_emb, val_emb = train_test_split(token_emb, test_size=0.1)

    # Hyperparameters
    input_dim = train_emb.shape[1]
    latent_dim = 32  # Adjust as needed
    learning_rate = 1e-4
    epochs = 2000

    # Model, optimizer, and loss function
    model = VAE(input_dim, latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Convert back to tensor for training
    train_emb = torch.from_numpy(train_emb).float()
    val_emb = torch.from_numpy(val_emb).float()

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        reconstructed_x, mu, log_var = model(train_emb)
        train_loss = vae_loss(reconstructed_x, train_emb, mu, log_var)
        train_loss.backward()
        optimizer.step()
        average_train_loss = train_loss.item() / len(train_emb)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            reconstructed_x_val, mu_val, log_var_val = model(val_emb)
            val_loss = vae_loss(reconstructed_x_val, val_emb, mu_val, log_var_val)
        average_val_loss = val_loss.item() / len(val_emb)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Average Train Loss: {average_train_loss}, Average Validation Loss: {average_val_loss}')

    # Use the trained encoder to compress the embeddings
    model.eval()
    with torch.no_grad():
        mu, _ = model.encoder(torch.from_numpy(token_emb).float())  # Use the full dataset if appropriate
        compressed_embeddings = mu.numpy()

    return compressed_embeddings

def get_fingerprint(token_emb):
    # Ensure tensor is on CPU and detached from the graph
    if isinstance(token_emb, torch.Tensor):
        token_emb = token_emb.detach().cpu().numpy()
    # A simple way to create a fingerprint of an array
    return hashlib.sha256(token_emb.tobytes()).hexdigest()