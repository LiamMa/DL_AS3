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

        self.encoder = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ELU(),
            # 32 x 16 x16
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ELU(),
            # 64 x 8 x 8
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ELU(),
            # 128 x 4 x 4
            Flatten(),
            nn.Linear(128*4*4, self.n_latent*2)
            # n_latent * 2
        )

        self.decoder = nn.Sequential(
            # n_latent
            nn.Linear(self.n_latent, 128 * 4 * 4),
            nn.ELU(),
            Reshape((128, 4, 4)),
            # 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ELU(),
            # 64 x 8 x 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ELU(),
            # 32 x 16 x 16
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            # 3 x 32 x 32
            nn.Tanh()
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
        outp = self.decoder(z)
        return outp

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        outp = self.decode(z)

        return outp, mu, logvar

    @staticmethod
    def loss_compute(X, y, mu, logvar):
        X, y = X.view(X.shape[0], -1), y.view(X.shape[0], -1)
        logpx_z = VAE.log_gaussian_distribution(X, y, torch.zeros_like(y))
        KL = 0.5 * torch.sum(1 + logvar - mu*mu - logvar.exp(), dim=1)
        ls = (-logpx_z - KL)
        return ls.mean()

    @staticmethod
    def log_gaussian_distribution(sample, mean, logvar, dim=1):
        """
        :param sample:   samples from gaussian, batch x latent_dim
        :param mean:     mean of each variable, batch x latent_dim
        :param logvar:   log of variance, log(sigma^2), batch x latent_dim
        :param dim:      sum over which dimension, mostly 1.
        :return:
        """
        log_p_sample = torch.sum(
            -0.5 * (np.log(2*np.pi) + logvar + (sample - mean) ** 2. * torch.exp(-logvar)),
            dim=dim)
        return log_p_sample


if __name__ == '__main__':
    vae = VAE(100)
    rand_x = torch.randn((16, 3, 32, 32))
    mu, logvar = vae.encode(rand_x)
    z = vae.reparam(mu, logvar)
    outp, logits = vae.decode(z)
    print(outp.shape, logits.shape)
