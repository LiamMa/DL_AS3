"""
The architecture is adapted from DCGAN
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from classify_svhn import Classifier
import numpy as np


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
        self.nc = 32
        self.encoder = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, self.nc, 4, 2, 1),
            nn.BatchNorm2d(self.nc),
            nn.ELU(),
            # 32 x 16 x16
            nn.Conv2d(self.nc, self.nc*2, 4, 2, 1),
            nn.BatchNorm2d(self.nc*2),
            nn.ELU(),
            # 64 x 8 x 8
            nn.Conv2d(self.nc*2, self.nc*4, 4, 2, 1),
            nn.BatchNorm2d(self.nc*4),
            nn.ELU(),
            # 128 x 4 x 4
            Flatten(),
            nn.Linear(self.nc*4*4*4, self.n_latent*2)
            # n_latent * 2
        )

        self.decoder = nn.Sequential(
            # n_latent
            Reshape((self.n_latent, 1, 1)),
            nn.ConvTranspose2d(self.n_latent, self.nc * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.nc * 4),
            nn.ELU(),
            # 128 x 4 x 4
            nn.ConvTranspose2d(self.nc*4, self.nc*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc * 2),
            nn.ELU(),
            # 64 x 8 x 8
            nn.ConvTranspose2d(self.nc*2, self.nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.nc),
            nn.ELU(),
            # 32 x 16 x 16
            nn.ConvTranspose2d(self.nc, 3, 4, 2, 1, bias=False),
            # 3 x 32 x 32
            nn.Tanh()
        )

    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu, logvar = torch.split(mu_logvar, self.n_latent, dim=1)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)

        z = mu + eps * std
        return z

    def decode(self, z):
        outp = self.decoder(z)
        return outp

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        outp = self.decode(z)

        return outp, mu, logvar

    @staticmethod
    def loss_compute(X, y, mu, logvar):
        recos_loss = F.mse_loss(y, X, reduction='sum')
        KL_loss = - 0.5 * torch.sum(1 + logvar - mu*mu - logvar.exp())
        ls = recos_loss + KL_loss
        return ls / X.shape[0]


if __name__ == '__main__':
    vae = VAE(100)
    rand_x = torch.randn((16, 3, 32, 32))
    mu, logvar = vae.encode(rand_x)
    z = vae.reparam(mu, logvar)
    outp, logits = vae.decode(z)
    print(outp.shape, logits.shape)
