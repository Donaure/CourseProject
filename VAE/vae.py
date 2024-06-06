import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def sample(self, mu, logvar):
        # reparameterize
        std_ = torch.exp(logvar * 0.5)
        z = mu + torch.randn_like(std_).to(mu.device) * std_
        return z

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return self.sample(mu, logvar)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sample(mu, logvar)
        return self.decoder(z), mu, logvar
