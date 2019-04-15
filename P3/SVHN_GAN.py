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


class GAN(nn.Module):

    def __init__(self, n_latent):
        super(GAN, self).__init__()

        self.n_latent = n_latent

        # self.generator = nn.Sequential(
        #     nn.Linear(self.n_latent, 128 * 4 * 4),
        #     nn.ELU(),
        #     Reshape((128, 4, 4)),
        #
        #     nn.Conv2d(128, 64, kernel_size=(5, 5), padding=(4, 4)),
        #     nn.ELU(),
        #
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
        #     nn.ELU(),
        #
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     nn.Conv2d(32, 16, kernel_size=(5, 5), padding=(4, 4)),
        #     nn.ELU(),
        #
        #     nn.Conv2d(16, 3, kernel_size=(5, 5), padding=(4, 4))
        # )

        self.generator = nn.Sequential(
            # n_latent
            nn.Linear(self.n_latent, 128 * 4 * 4),
            nn.ELU(),
            Reshape((128, 4, 4)),
            # 128 x 4 x 4
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # 64 x 8 x 8
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # 32 x 16 x 16
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
            # 3 x 32 x 32
        )

        self.discriminator = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ELU(),
            # 32 x 16 x16
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # 64 x 8 x 8
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            # 128 x 4 x 4
            nn.Conv2d(128, 1, 4, 1, 0),
            Flatten(),
            nn.Sigmoid()
            # 1
        )

        for nm, w in self.generator.named_parameters():
            if 'bias' not in nm:
                nn.init.normal_(w, 0, 0.02)
        for nm, w in self.discriminator.named_parameters():
            if 'bias' not in nm:
                nn.init.normal_(w, 0, 0.02)

    def forward(self, rand_x):
        pass

    def generate(self, rand_x):
        return self.generator(rand_x)

    def discriminate(self, x):
        return self.discriminator(x)

    def gradient_penalty(self, real_data, fake_data):
        batch_size = real_data.shape[0]

        eps = torch.rand(batch_size, 1, 1, 1).to(real_data.device)
        x_hat = eps * real_data + (1-eps) * fake_data
        x_hat.requires_grad = True

        res_D = self.discriminator(x_hat)
        grad_x_hat = torch.autograd.grad(res_D, x_hat, grad_outputs=torch.ones(res_D.shape, device=real_data.device),
                                         create_graph=True)[0].view(res_D.size(0), -1)

        grad_loss = torch.pow((grad_x_hat.norm(2, dim=1) - 1), 2).mean()

        return grad_loss


if __name__ == '__main__':
    GAN(100)