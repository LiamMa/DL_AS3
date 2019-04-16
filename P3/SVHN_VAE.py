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


if __name__ == '__main__':
    vae = VAE(100)
    rand_x = torch.randn((16, 3, 32, 32))
    mu, logvar = vae.encode(rand_x)
    z = vae.reparam(mu, logvar)
    outp, logits = vae.decode(z)
    print(outp.shape, logits.shape)
