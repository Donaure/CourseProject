import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dim=2, h1_dim=512, h2_dim=256, input_dim=784):
        super(Decoder, self).__init__()
        # 4 layer
        # self.fc1 = nn.Linear(latent_dim, h2_dim)
        # self.fc2 = nn.Linear(h2_dim, h1_dim)
        # self.fc3 = nn.Linear(h1_dim, 1024)
        # self.fc4 = nn.Linear(1024, input_dim)

        # 3 layer
        self.fc1 = nn.Linear(latent_dim, h2_dim)
        self.fc2 = nn.Linear(h2_dim, h1_dim)
        self.fc3 = nn.Linear(h1_dim, input_dim)

    def forward(self, z):
        # 4 layer
        # z = F.relu(self.fc1(z))
        # z = F.relu(self.fc2(z))
        # z = F.relu(self.fc3(z))
        # z = torch.sigmoid(self.fc4(z))

        # 3 layer
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = torch.sigmoid(self.fc3(z))

        # z.reshape(-1, 28, 28)
        return z
