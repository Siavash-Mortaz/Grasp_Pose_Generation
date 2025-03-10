import torch
import torch.nn as nn
import torch.nn.functional as F
class CVAE_02_4(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim, num_gaussians=5):
        super(CVAE_02_4, self).__init__()
        self.num_gaussians = num_gaussians
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            torch.nn.SiLU(),
            nn.Dropout(0.1),  # Dropout added
            nn.Linear(1024, 512),
            torch.nn.SiLU(),
            nn.Dropout(0.1),  # Dropout added
            nn.Linear(512, 256),
            torch.nn.SiLU(),
            nn.Dropout(0.1),  # Dropout added
            nn.Linear(256, 128),
            torch.nn.SiLU(),
            nn.Linear(128, latent_dim * num_gaussians * 2)
        )

        # Mixing coefficients for MoG
        self.mixing_coefficients = nn.Linear(128, num_gaussians)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            torch.nn.SiLU(),
            nn.Dropout(0.1),  # Dropout added
            nn.Linear(128, 256),
            torch.nn.SiLU(),
            nn.Dropout(0.1),  # Dropout added
            nn.Linear(256, 512),
            torch.nn.SiLU(),
            nn.Dropout(0.1),  # Dropout added
            nn.Linear(512, 1024),
            torch.nn.SiLU(),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)  # Pass through encoder

        # Compute mixing coefficients from the last hidden layer BEFORE splitting
        logits = self.mixing_coefficients(h[:, :128])  # Use only the last hidden layer (128D)
        mixing_coeffs = F.softmax(logits, dim=-1)  # Convert logits to probabilities

        # Compute Mean & Log Variance
        mean, log_var = torch.chunk(h, 2, dim=-1)  # Split into two parts (each of size `latent_dim * num_gaussians`)

        # Reshape mean and log_var to match (batch_size, num_gaussians, latent_dim)
        mean = mean.view(-1, self.num_gaussians, self.latent_dim)
        log_var = log_var.view(-1, self.num_gaussians, self.latent_dim)

        return mean, log_var, mixing_coeffs

    def reparameterize(self, mean, log_var, mixing_coeffs):
        """
        Sample from the MoG latent space using GMM sampling.
        """
        std = torch.exp(0.5 * log_var)

        # Sample a component index from the mixture
        component = torch.multinomial(mixing_coeffs, 1).squeeze(1)  # Sample which Gaussian to use
        batch_indices = torch.arange(mean.shape[0])

        # Select the corresponding mean and variance for the chosen component
        selected_mean = mean[batch_indices, component, :]
        selected_std = std[batch_indices, component, :]

        # Standard reparameterization trick
        eps = torch.randn_like(selected_std)
        return selected_mean + eps * selected_std

    def decode(self, z, c):
        combined = torch.cat([z, c], dim=-1)
        return self.decoder(combined)

    def forward(self, x, c):
        mean, log_var, mixing_coeffs = self.encode(x)
        z = self.reparameterize(mean, log_var, mixing_coeffs)
        return self.decode(z, c), mean, log_var, mixing_coeffs