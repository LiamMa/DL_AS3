import torch
from torch.utils.data import DataLoader
from binary_mnist import get_dataset
from VAE import VAE
import time
import logging
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 20
TEST_INTERVAL = 5
N_LATENT = 100
LR = 3e-4


def train(model, train_iter, test_iter, optimizer, num_epochs, test_interval):
    model.train()
    print('Training begins.')
    all_loss = []
    for epoch in range(1, num_epochs+1):
        epoch_loss = .0
        n_batches = 0
        start = time.time()
        for batch, X in enumerate(train_iter):
            n_batches += 1
            X = X[0].to(dev)
            optimizer.zero_grad()

            y, mu, logvar = model(X)
            loss = model.loss_compute(X, y, mu, logvar)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= n_batches
        all_loss.append(epoch_loss)
        print('Epoch {}, Loss: {:.4f}, ELBO: {:.4f}, Time: {:.4f}'.format(
            epoch, epoch_loss, -epoch_loss, time.time()-start))

        if epoch % test_interval == 0:
            test(model, test_iter, epoch)

    return all_loss


def test(model, test_iter, epoch):
    model.eval()
    start = time.time()

    epoch_loss = .0
    n_batches = 0
    with torch.no_grad():
        for batch, X in enumerate(test_iter):
            n_batches += 1
            X = X[0].to(dev)
            y, mu, logvar = model(X)
            loss = model.loss_compute(X, y, mu, logvar)

            epoch_loss += loss.item()
        epoch_loss /= n_batches
    print('----------Test, Loss: {:.4f}, ELBO: {:.4f}, Time: {:.4f}----------'.format(
        epoch_loss, -epoch_loss, time.time()-start))

    # generate images
    sample_x = X[:16].cpu().numpy().reshape(4, 4, 28, 28)
    recons_x = y[:16].cpu().numpy().reshape(4, 4, 28, 28)

    canvas1 = np.zeros((4 * 28, 4 * 28))
    canvas2 = np.zeros((4 * 28, 4 * 28))
    for i in range(4):
        for j in range(4):
            canvas1[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = sample_x[i, j]
            canvas2[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = recons_x[i, j]

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(canvas1, cmap='Greys')
    axes[0].set_title('original')
    axes[0].axis('off')
    axes[1].imshow(canvas2, cmap='Greys')
    axes[1].set_title('reconstructed')
    axes[0].axis('off')
    plt.savefig('Epoch_{}_test_samples.png'.format(epoch))

def evaluate_LLE(model, x, z):
# def test(model, test_iter):
#     model.eval()
    pass

def main():
    # load dataset and data_iter
    train_dataset = get_dataset('train')
    valid_dataset = get_dataset('valid')
    test_dataset  = get_dataset('test')

    train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_iter = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # test_iter  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # load model, loss, optimizer
    model = VAE(n_latent=N_LATENT)
    model.to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # train
    train(model, train_iter, valid_iter, optimizer, NUM_EPOCHS, TEST_INTERVAL)

if __name__ == '__main__':
    main()