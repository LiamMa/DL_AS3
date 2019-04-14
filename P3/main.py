from classify_svhn import get_data_loader
import torch
import torch.nn as nn
import time
from SVHN_GAN import GAN
from SVHN_VAE import VAE

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 20
TEST_INTERVAL = 5
N_LATENT = 100
LR_G = 3e-4
LR_D = 3e-4


def train_gan(model, train_iter, test_iter, num_epochs, G_update_iterval, test_interval):
    model.train()
    all_loss_g = []
    all_loss_d = []

    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(model.generator.parameters(), lr=LR_G)
    optimizerD = torch.optim.Adam(model.discriminator.parameters(), lr=LR_D)

    for epoch in range(1, num_epochs+1):
        epoch_ls_g = []
        epoch_ls_d = []
        epoch_ce_d = []
        start = time.time()
        for batch, (X, y) in enumerate(train_iter):
            X = X.to(dev)
            X_label = torch.ones(X.shape[0]).to(dev)
            noise = torch.rand(size=(X.shape[0], N_LATENT)).to(dev)
            fake_label = torch.zeros(X.shape[0]).to(dev)

            # update discriminator
            optimizerD.zero_grad()
            X_fake = model.generator(noise)

            outp_real = model.discriminator(X)
            outp_fake = model.discriminator(X_fake.detach())
            gradient_penalty = model.gradient_penalty(X, X_fake.detach())
            CELoss = criterion(outp_real, X_label) + criterion(outp_fake, fake_label)
            d_loss = CELoss + gradient_penalty
            d_loss.backward()
            optimizerD.step()
            epoch_ls_d.append(d_loss.item())
            epoch_ce_d.append(CELoss.item())
            # update generator
            if (batch+1) % G_update_iterval == 0:
                optimizerG.zero_grad()
                outp_fake = model.discriminator(X_fake)
                fake_label = torch.ones(X.shape[0]).to(dev)
                g_loss = criterion(outp_fake, fake_label)
                g_loss.backward()
                optimizerG.step()
                epoch_ls_g.append(g_loss.item())

        all_loss_g.append(sum(epoch_ls_g) / len(epoch_ls_g))
        all_loss_d.append(sum(epoch_ls_d) / len(epoch_ls_d))
        print('Epoch {},\tGenerator Loss: {:.4f},\tDiscriminator Loss: {:.4f},\tDsicriminator CELoss: {:.4f},\tTime: {:.2f}.'.format(
            epoch, all_loss_g[-1], all_loss_d[-1], sum(epoch_ce_d) / len(epoch_ce_d), time.time() - start)
        )

        if epoch % test_interval == 0:
            test_gan(model, test_iter, epoch)
            model.train()

def test_gan(model, test_iter, epoch):
    # model.eval()
    # epoch_ls_g = []
    # epoch_ls_d = []
    # start = time.time()
    #
    # for batch, (X, y) in enumerate(test_iter):
    #     pass
    pass




def main():
    train_iter, valid_iter, test_iter = get_data_loader('data', BATCH_SIZE)
    model_gan = GAN(N_LATENT)
    model_gan.to(dev)
    train_gan(model_gan, train_iter, valid_iter, NUM_EPOCHS, G_update_iterval=2, test_interval=5)

if __name__ == '__main__':
    main()