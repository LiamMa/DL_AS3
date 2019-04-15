from classify_svhn import get_data_loader
import torch
import torch.nn as nn
import time
from SVHN_GAN import GAN
from SVHN_VAE import VAE
from torchvision.utils import save_image
from torch.nn import functional
import numpy as np

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 50
TEST_INTERVAL = 5
N_LATENT = 100
LR_G = 0.0001
LR_D = 0.001
beta1 = 0.5


def train_gan(model, train_iter, test_iter, num_epochs, G_update_iterval, test_interval):
    model.train()
    all_loss_g = []
    all_loss_d = []
    fixed_noise = torch.randn(64, N_LATENT).to(dev)

    criterion = nn.BCELoss()
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
            one_label = torch.ones((X.shape[0],), device=dev)
            zero_label = torch.zeros((X.shape[0],), device=dev)

            # update discriminator
            optimizerD.zero_grad()

            outp_real = model.discriminator(X).view(-1)
            errD_real = criterion(outp_real, one_label)
            outp_fake = model.discriminator(X_fake.detach()).view(-1)
            errD_fake = criterion(outp_fake, zero_label)

            D_X = outp_real.mean().item()
            D_G_z1 = outp_fake.mean().item()

            gradient_penalty = 0 # model.gradient_penalty(X, X_fake.detach())
            CELoss = errD_real + errD_fake
            d_loss = CELoss + gradient_penalty
            d_loss.backward()
            optimizerD.step()
            epoch_ls_d.append(d_loss.item())
            epoch_ce_d.append(CELoss.item())

            # update generator
            if (batch+1) % G_update_iterval == 0:
                optimizerG.zero_grad()
                outp_fake = model.discriminator(X_fake).view(-1)
                errG = criterion(outp_fake, one_label)

                D_G_z2 = outp_fake.mean().item()
                g_loss = errG
                g_loss.backward()
                optimizerG.step()
                epoch_ls_g.append(g_loss.item())

        all_loss_g.append(sum(epoch_ls_g) / len(epoch_ls_g))
        all_loss_d.append(sum(epoch_ls_d) / len(epoch_ls_d))
        template = 'Epoch {}, G Loss:{:.3f}, D Loss:{:.3f}, D CELoss:{:.3f}, ' \
                   'D(x):{:.2f}, D(G(z))_1:{:.2f}, D(G(z))_2:{:.2f}, Time: {:.2f}.'
        print(template.format(epoch, all_loss_g[-1], all_loss_d[-1], sum(epoch_ce_d) / len(epoch_ce_d),
                              D_X, D_G_z1, D_G_z2, time.time() - start)
        )

        if epoch % test_interval == 0:
            test_gan(model, test_iter, fixed_noise, epoch)
            model.train()


def test_gan(model, test_iter, fixed_noise, epoch):
    # model.eval()
    # epoch_ls_g = []
    # epoch_ls_d = []
    # start = time.time()
    #
    # for batch, (X, y) in enumerate(test_iter):
    #     pass
    def deprocess(x):
        return np.uint8((x + 1) / 2 * 255)
    with torch.no_grad():
        X_fake = model.generator(fixed_noise).cpu()
        X_fake = (X_fake+1)/2
    save_image(X_fake, 'GAN_Epoch{}.png'.format(epoch))




def main():
    train_iter, valid_iter, test_iter = get_data_loader('data', BATCH_SIZE)
    model_gan = GAN(N_LATENT)
    model_gan.to(dev)
    train_gan(model_gan, train_iter, valid_iter, NUM_EPOCHS, G_update_iterval=1, test_interval=4)

if __name__ == '__main__':
    main()