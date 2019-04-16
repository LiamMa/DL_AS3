import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

import os
import time
import logging
from classify_svhn import get_data_loader
from SVHN_GAN import GAN
from SVHN_VAE import VAE
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CHECKPOINT_PATH = 'ckpt'
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_EPOCHS = 100
TEST_INTERVAL = 5
N_LATENT = 100
LR_G = 0.0001
LR_D = 0.001
beta1 = 0.5
LAMBDA = 10


def train_gan(model, train_iter, test_iter, num_epochs, G_update_iterval, test_interval):
    model.train()
    all_loss_g = []
    all_loss_d = []
    fixed_noise = torch.randn(64, N_LATENT).to(dev)

    optimizerG = torch.optim.Adam(model.generator.parameters(), lr=LR_G, betas=(beta1, 0.999))
    optimizerD = torch.optim.Adam(model.discriminator.parameters(), lr=LR_D, betas=(beta1, 0.999))

    for epoch in range(1, num_epochs+1):
        epoch_ls_g = []
        epoch_ls_d = []
        epoch_ce_d = []
        start = time.time()
        for batch, (X, y) in enumerate(train_iter):

            X = X.to(dev)
            noise = torch.randn(size=(X.shape[0], N_LATENT)).to(dev)
            X_fake = model.generator(noise)

            # update discriminator
            optimizerD.zero_grad()

            outp_real = model.discriminator(X).view(-1)
            outp_fake = model.discriminator(X_fake.detach()).view(-1)
            gradient_penalty = model.gradient_penalty(X, X_fake.detach())
            wd_distance = outp_real.mean() - outp_fake.mean()
            d_loss = -wd_distance + LAMBDA * gradient_penalty
            # D_X = outp_real.mean().item()
            # D_G_z1 = outp_fake.mean().item()

            d_loss.backward()
            optimizerD.step()
            epoch_ls_d.append(d_loss.item())

            # update generator
            if (batch+1) % G_update_iterval == 0:
                optimizerG.zero_grad()
                noise = torch.randn(size=(X.shape[0], N_LATENT)).to(dev)
                X_fake = model.generator(noise)
                outp_fake = model.discriminator(X_fake)
                g_loss = -outp_fake.mean()
                # outp_fake = model.discriminator(X_fake).view(-1)
                # errG = criterion(outp_fake, one_label)

                # D_G_z2 = outp_fake.mean().item()
                # g_loss = errG
                g_loss.backward()
                optimizerG.step()
                epoch_ls_g.append(g_loss.item())

        all_loss_g.append(sum(epoch_ls_g) / len(epoch_ls_g))
        all_loss_d.append(sum(epoch_ls_d) / len(epoch_ls_d))
        template = 'Epoch {}, G Loss:{:.2f}, D Loss:{:.2f}, G WD:{:.2f}, ' \
                   'Time: {:.2f}.'
        logger.info(template.format(epoch, all_loss_g[-1], all_loss_d[-1],
                              wd_distance, time.time() - start)
        )

        if epoch % test_interval == 0:
            model.eval()
            test_generate_imgs(model.generator, fixed_noise, 'GAN_generated_samples_epoch_{}.png'.format(epoch))
            test_all(model.generator, 4, N_LATENT)
            model.train()


def test_generate_imgs(netG, fixed_noise, save_name):
    with torch.no_grad():
        x = netG(fixed_noise).cpu()
        x = utils.de_normalize(x)
    save_image(x, save_name)


def test_disentangle(netG, batch_size, n_latent, eps, model_name='GAN'):

    z = torch.randn(size=(batch_size, n_latent), device=next(netG.parameters()).device)
    res = []
    for i in range(n_latent):
        with torch.no_grad():
            z_hat = z + 0
            z_hat[:, i] += eps
            x_hat = netG(z_hat).cpu()
        res.append(x_hat)

        if len(res) == 10:
            all_imgs = torch.stack(res, dim=1)
            all_imgs = all_imgs.permute([0, 1, 3, 4, 2])
            utils.make_grid_img(all_imgs, '{}_disentangle_{:.3f}_{}-{}-dim.png'.format(model_name, eps, i-10, i))
            res = []


def test_interpolate(netG, batch_size, n_latent, model_name='GAN'):

    z0 = torch.randn(size=(batch_size, n_latent), device=next(netG.parameters()).device)
    z1 = torch.randn(size=(batch_size, n_latent), device=next(netG.parameters()).device)
    alpha = [0+0.1*i for i in range(11)]

    # interpolate 1: interpolate in latent space z.
    res = []
    for a in alpha:
        z = a * z0 + (1-a) * z1
        with torch.no_grad():
            x = netG(z).cpu()
        res.append(x)
    all_imgs = torch.stack(res, dim=1)
    all_imgs = all_imgs.permute([0, 1, 3, 4, 2])
    utils.make_grid_img(all_imgs, '{}_interpolate1_latent.png'.format(model_name))

    # interpolate data space:
    res = []
    with torch.no_grad():
        x0 = netG(z0).cpu()
        x1 = netG(z1).cpu()
    for a in alpha:
        x = a * x0 + (1-a) * x1
        res.append(x)
    all_imgs = torch.stack(res, dim=1)
    all_imgs = all_imgs.permute([0, 1, 3, 4, 2])
    utils.make_grid_img(all_imgs, '{}_interpolate2_data.png'.format(model_name))


def test_all(netG, batch_size, n_latent, model_name='GAN'):
    test_disentangle(netG, batch_size, n_latent, 1e-1, model_name)
    test_interpolate(netG, batch_size, n_latent, model_name)


def main():
    train_iter, valid_iter, test_iter = get_data_loader('data', BATCH_SIZE)
    model_gan = GAN(N_LATENT)
    model_gan.to(dev)
    train_gan(model_gan, train_iter, valid_iter, NUM_EPOCHS, G_update_iterval=1, test_interval=4)


if __name__ == '__main__':
    main()