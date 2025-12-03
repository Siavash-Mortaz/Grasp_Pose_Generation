"""Conditional Variational Autoencoder (CVAE) model definitions."""

import torch
import torch.nn as nn


# =================== MODEL 01 - CVAE_01 ==============================
# WITH TOTAL 3 LAYERS FOR EACH ENCODER AND DECODER
# 2 LINEAR LAYERS AND 1 RELU LAYER FOR EACH ENCODER AND DECODER
class CVAE_01(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE_01, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def encode(self, x, c):
        combined = torch.cat([x, c], dim=-1)
        h = self.encoder(combined)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        return mean, log_var

    def decode(self, z, c):
        combined = torch.cat([z, c], dim=-1)
        return self.decoder(combined)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, c):
        mean, log_var = self.encode(x, c)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, c), mean, log_var


# =================== MODEL 02 - CVAE_02 ==============================
# ADD ONE MORE LAYER
# WITH TOTAL 5 LAYERS FOR EACH ENCODER AND DECODER
# 3 LINEAR LAYERS AND 2 RELU LAYERS FOR EACH ENCODER AND DECODER
class CVAE_02(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE_02, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, x, c):
        combined = torch.cat([x, c], dim=-1)
        h = self.encoder(combined)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        return mean, log_var

    def decode(self, z, c):
        combined = torch.cat([z, c], dim=-1)
        return self.decoder(combined)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, c):
        mean, log_var = self.encode(x, c)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, c), mean, log_var


# =================== MODEL 03 - CVAE_03 ==============================
# ADD DROPOUT LAYER (dropout layers can help prevent overfitting)
# TOTAL 9 LAYERS FOR EACH ENCODER AND DECODER
# 3 LINEAR LAYERS, 2 BATCH NORMALIZATION LAYERS, 2 RELU LAYERS, AND 2 DROUPOUT LAYERS FOR EACH ENCODER AND DECODER
class CVAE_03(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE_03, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, input_dim)
        )

    def encode(self, x, c):
        combined = torch.cat([x, c], dim=-1)
        h = self.encoder(combined)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        return mean, log_var

    def decode(self, z, c):
        combined = torch.cat([z, c], dim=-1)
        return self.decoder(combined)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, c):
        mean, log_var = self.encode(x, c)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, c), mean, log_var


# =================== MODEL 04 - CVAE_02_1 ==============================
# ADD MORE LAYERS TO CVAE_02
# WITH TOTAL 7 LAYERS FOR EACH ENCODER AND DECODER
# 4 LINEAR LAYERS AND 3 RELU LAYERS FOR EACH ENCODER AND DECODER
class CVAE_02_1(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE_02_1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x, c):
        combined = torch.cat([x, c], dim=-1)
        h = self.encoder(combined)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        return mean, log_var

    def decode(self, z, c):
        combined = torch.cat([z, c], dim=-1)
        return self.decoder(combined)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, c):
        mean, log_var = self.encode(x, c)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, c), mean, log_var


# =================== MODEL 05 - CVAE_02_2 ==============================
# ADD MORE LAYERS TO CVAE_02_1
# WITH TOTAL 9 LAYERS FOR EACH ENCODER AND DECODER
# 5 LINEAR LAYERS AND 4 RELU LAYERS FOR EACH ENCODER AND DECODER
class CVAE_02_2(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE_02_2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Mean and log-variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x, c):
        combined = torch.cat([x, c], dim=-1)
        h = self.encoder(combined)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        return mean, log_var

    def decode(self, z, c):
        combined = torch.cat([z, c], dim=-1)
        return self.decoder(combined)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, c):
        mean, log_var = self.encode(x, c)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, c), mean, log_var


# =================== MODEL 06 - CVAE_02_3 ==============================
# REMOVE OBJECT (CONDITION) FROM ENCODER IN CVAE02_2
# WITH TOTAL 9 LAYERS FOR EACH ENCODER AND DECODER
# 5 LINEAR LAYERS AND 4 RELU LAYERS FOR EACH ENCODER AND DECODER
class CVAE_02_3(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE_02_3, self).__init__()
        # Encoder: Takes only the hand pose as input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # Mean and log-variance
        )
        # Decoder: Takes latent vector and condition as input
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=-1)
        return mean, log_var

    def decode(self, z, c):
        combined = torch.cat([z, c], dim=-1)
        return self.decoder(combined)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, c):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z, c), mean, log_var

