import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import matplotlib

import os
import time
import logging
from classify_svhn import get_data_loader
from SVHN_GAN import GAN
from SVHN_VAE import VAE
import utils
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser('P3')
parser.add_argument('--model', type=str, default='VAE',
                    help='run vae or gan.')
parser.add_argument('--test_interval', type=int, default=10,
                    help='epoch intervals to test, e.g., logpx')
parser.add_argument('--num_epochs', type=int, default=20,
                    help='how many epochs to run.')
parser.add_argument('--batch_size', type=int, default=256,
                    help='batch size')
parser.add_argument('--G_update_interval', type=int, default=1,
                    help='how many batches before generator updates once.')
parser.add_argument('--save_model', action='store_true',
                    help='save model after training')
parser.add_argument('--load_model', action='store_true',
                    help='load model to test')
parser.add_argument('--mode', type=str, default='train',
                    help='train: train a model, '
                         'test: test a trained model for qualitative analysis,'
                         'gen: use a trained model to generate 1000 images.')
args = parser.parse_args()

CHECKPOINT_PATH = 'ckpt'
SAMPLE_PATH = '{}/samples'.format(args.model)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_LATENT = 100
LR_G = 0.0001
LR_D = 0.001
LR_VAE = 1e-3
beta1 = 0.5
LAMBDA = 10


def train_vae(model, train_iter, test_iter, num_epochs, test_interval, save_model=False):
    test_vae(model, test_iter, 0)
    logger.info('Training VAE begins.')
    all_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_VAE)  #, betas=(beta1, 0.999))
    for epoch in range(1, num_epochs+1):
        model.train()
        epoch_loss = .0
        n_batches = 0
        n_images = 0
        start = time.time()
        for batch, data in enumerate(train_iter):
            X = data[0].to(dev)
            n_batches += 1
            n_images += X.shape[0]
            optimizer.zero_grad()

            y, mu, logvar = model(X)
            loss = model.loss_compute(X, y, mu, logvar)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches
        all_loss.append(epoch_loss)
        logger.info('Epoch {}, Loss: {:.4f}, ELBO: {:.4f}, Time: {:.4f}, Img processed: {}'.format(
            epoch, epoch_loss, -epoch_loss, time.time()-start, n_images))
        if epoch % test_interval == 0:
            test_vae(model, test_iter, epoch)

    if save_model:
        utils.model_save(model, 'P3_VAE.pt')
    utils.make_plot(all_loss, save_name='P3_vae_learning_curve.png', title='VAE Learning Curve', tickets=['Epochs', 'Loss'])
    return all_loss


def test_vae(model, test_iter, epoch):
    model.eval()
    start = time.time()

    sum_loss = 0.0
    n_examples = 0
    with torch.no_grad():
        for batch, data in enumerate(test_iter):
            X = data[0].to(dev)
            y, mu, logvar = model(X)
            loss = model.loss_compute(X, y, mu, logvar)

            sum_loss += loss.item() * X.shape[0]
            n_examples += X.shape[0]
        avg_loss = sum_loss / n_examples
    logger.info('Test, Avg Loss: {:.4f}, Avg ELBO: {:.4f}, Time: {:.4f}'.format(
        avg_loss, -avg_loss, time.time()-start))

    # generate images
    sample_x = X[:16].view(4, 4, 3, 32, 32).contiguous()
    recons_x = y[:16].view(4, 4, 3, 32, 32).contiguous()
    all_x = torch.cat((sample_x, recons_x), dim=1).permute([0, 1, 3, 4, 2]).cpu().numpy()
    utils.make_grid_img(all_x, 'P3_VAE_Epoch_{}_test_samples.png'.format(epoch),
                        title='Left 4: sample, Right 4: reconstruction')
    # for i in range(4):
    #     for j in range(4):
    #         canvas1[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = sample_x[i, j]
    #         canvas2[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons_x[i, j]
    #
    # fig, axes = plt.subplots(1, 2)
    # axes[0].imshow(canvas1, cmap='Greys')
    # axes[0].set_title('original')
    # axes[0].axis('off')
    # axes[1].imshow(canvas2, cmap='Greys')
    # axes[1].set_title('reconstructed')
    # axes[0].axis('off')
    # plt.savefig('P3_VAE_Epoch_{}_test_samples.png'.format(epoch))


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


def test_generate_imgs(netG, z, model_name, origin_img=None):
    save_name = 'P3_Quality_1_{}.png'.format(model_name)
    with torch.no_grad():
        x = netG(z).cpu()
        x = utils.de_normalize(x).view(2, x.shape[0]//2, *(x.shape[1:]))
        x = x.permute([0, 1, 3, 4, 2])
    utils.make_grid_img(x, save_name)
    if origin_img is not None:
        origin_img = origin_img.cpu().view(2, origin_img.shape[0]//2, *(origin_img.shape[1:])).permute([0, 1, 3, 4, 2])
        save_name = 'P3_Quality_1_{}_origin.png'.format(model_name)
        utils.make_grid_img(origin_img, save_name)


def test_disentangle(netG, z, eps, model_name):

    res = []
    for i in range(z.shape[1]):
        with torch.no_grad():
            z_hat = z + 0
            z_hat[:, i] += eps
            x_hat = netG(z_hat).cpu()
        res.append(x_hat)

        if len(res) == 10:
            all_imgs = torch.stack(res, dim=1)
            all_imgs = all_imgs.permute([0, 1, 3, 4, 2])
            utils.make_grid_img(all_imgs, 'P3_Quality_2_{}_disentangle_{:.3f}_{}-{}-dim.png'.format(model_name, eps, i-10, i))
            res = []


def test_interpolate(netG, z0, z1, model_name):

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
    utils.make_grid_img(all_imgs, 'P3_Quality_3_{}_interpolate1_latent.png'.format(model_name))

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
    utils.make_grid_img(all_imgs, 'P3_Quality_3_{}_interpolate1_latent.png'.format(model_name))


def test_all(model, test_iter, batch_size, n_latent, model_name):
    model.eval()
    if model_name == 'GAN':
        z0 = torch.randn(size=(batch_size, n_latent), device=dev)
        z1 = torch.randn(size=(batch_size, n_latent), device=dev)
        netG = model.generator
    else:
        x, z = [], []
        for num_batch, (X, y) in enumerate(test_iter):
            X = X.to(dev)
            x.append(X)
            z.append(model.reparam(*(model.encode(X))))
            if num_batch == 1:
                break
        z0, z1 = z
        netG = model.decoder

    # qualitative analysis 1
    test_generate_imgs(netG, torch.cat([z0, z1]), model_name, origin_img=torch.cat(x))
    # qualitative analysis 2
    test_disentangle(netG, z0, 1e-1, model_name)
    # qualitative analysis 3
    test_interpolate(netG, z0, z1, model_name)


def generate_images(netG, latents, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    netG.eval()
    with torch.no_grad():
        reconstruction = netG(latents).tanh()
        if isinstance(reconstruction, tuple):
            reconstruction = reconstruction[0]
        reconstruction = reconstruction.tanh().permute([0, 2, 3, 1]).cpu().numpy()

    reconstruction = utils.de_normalize(reconstruction)
    for i in range(reconstruction.shape[0]):
        matplotlib.image.imsave(os.path.join(save_path, '{}.png'.format(i)), reconstruction[i])


def main(args):
    train_iter, valid_iter, test_iter = get_data_loader('data', args.batch_size)
    # ---------- Try to load model first ----------
    model = None
    if args.load_model:
        try:
            model = utils.model_load('P3_{}.pt'.format(args.model))
        except:
            logger.info('Loading model failed, will train the model first.')
            args.load_model = False
    # ---------- GAN related ----------
    if args.model == 'GAN':
        if not args.load_model or args.mode == 'train':
            model = GAN(N_LATENT)
            model.to(dev)
            train_gan(model, train_iter, test_iter, args.num_epochs,
                      G_update_iterval=args.G_update_interval, test_interval=args.test_interval,
                      save_model=args.save_model)
    # ---------- VAE related ----------
    else:
        if args.mode == 'train' or not args.load_model:
            model = VAE(N_LATENT)
            model.to(dev)
            train_vae(model, train_iter, test_iter, args.num_epochs,
                      test_interval=args.test_interval, save_model=args.save_model)
        model.eval()

        if args.mode == 'test':
            test_all(model, test_iter, args.batch_size, N_LATENT, model_name='VAE')

        if args.mode == 'gen':
            n_imgs = 0
            latent = []
            with torch.no_grad():
                for batch in test_iter:
                    X = batch[0].to(dev)
                    n_imgs += X.shape[0]
                    latent.append(model.reparam(*model.encode(X)))

                    if n_imgs > 1000:
                        break
                latent = torch.cat(latent, dim=0)[:1000]
            generate_images(model.decoder, latent, save_path='sample/VAE/samples')


if __name__ == '__main__':
    main(args)
