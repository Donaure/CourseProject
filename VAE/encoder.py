import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim=784, h1_dim=512, h2_dim=256, latent_dim=2):
        super(Encoder, self).__init__()
        self.input_dim = input_dim

        # 4 layer
        # self.fc0 = nn.Linear(input_dim, 1024)
        # self.fc1 = nn.Linear(1024, h1_dim)
        # self.fc2 = nn.Linear(h1_dim, h2_dim)
        # self.mu = nn.Linear(h2_dim, latent_dim)
        # self.logvar = nn.Linear(h2_dim, latent_dim)

        # 3 layer
        self.fc1 = nn.Linear(input_dim, h1_dim)
        self.fc2 = nn.Linear(h1_dim, h2_dim)
        self.mu = nn.Linear(h2_dim, latent_dim)
        self.logvar = nn.Linear(h2_dim, latent_dim)

    def forward(self, x):
        # 4 layer
        # x = self.fc0(x.view(-1, self.input_dim))
        # x = F.relu(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))

        # 3 layer
        x = self.fc1(x.view(-1, self.input_dim))
        x = F.relu(x)
        x = F.relu(self.fc2(x))

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar
