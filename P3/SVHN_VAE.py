import torch
import torch.nn as nn
import torch.nn.functional as F
from classify_svhn import Classifier

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, inp):
        return inp.view((inp.shape[0], -1))


class Reshape(nn.Module):

    def __init__(self, tgt_shape):
        super(Reshape, self).__init__()
        self.tgt_shape = tgt_shape

    def forward(self, inp):
        return inp.view([inp.shape[0], *self.tgt_shape])


class VAE(nn.Module):

    def __init__(self, n_latent=100):
        super(VAE, self).__init__()
        self.n_latent = n_latent

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 256, kernel_size=(5, 5)),
            nn.ELU(),

            Flatten(),
            nn.Linear(256, self.n_latent*2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, 256),
            nn.ELU(),
            Reshape((256, 1, 1)),

            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4)),
            nn.ELU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),

            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=(2, 2))
        )

    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = torch.split(mu_logvar, self.n_latent, dim=1)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(mu)

        z = mu + eps * std
        return z

    def decode(self, z):
        logits = self.decoder(z)
        outp = torch.sigmoid(logits)
        return outp, logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        outp, logits = self.decode(z)

        return outp, mu, logvar

    @staticmethod
    def loss_compute(X, y, mu, logvar):
        X, y = X.view(-1, 784), y.view(-1, 784)
        logpx_z = -torch.sum(X*torch.log(y+1e-10)+ (1-X)*torch.log(1-y+1e-10), dim=1)
        KL = 0.5 * torch.sum(1 + logvar - mu*mu - logvar.exp(), dim=1)
        ls = (logpx_z - KL)
        return ls.mean()
