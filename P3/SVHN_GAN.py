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

        self.generator = nn.Sequential(
            nn.Linear(self.n_latent, 256),
            nn.ELU(),
            Reshape((256, 1, 1)),

            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=(4, 4)),
            nn.ELU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=(2, 2)),
            nn.ELU(),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=(5, 5), padding=(4, 4)),
            nn.ELU(),

            nn.Conv2d(16, 3, kernel_size=(5, 5), padding=(4, 4))
        )


        self.discriminator = Classifier()

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
        x_hat = x_hat.view(real_data.shape)
        x_hat.requires_grad = True

        res_D = self.discriminator(x_hat)
        grad_x_hat = torch.autograd.grad(res_D, x_hat, grad_outputs=torch.ones(res_D.shape, device=real_data.device),
                                         create_graph=True)[0].view(res_D.size(0), -1)

        grad_loss = torch.pow((grad_x_hat.norm(2, dim=1) - 1), 2).mean()

        return grad_loss