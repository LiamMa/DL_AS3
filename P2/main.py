import torch
from torch.utils.data import DataLoader
from binary_mnist import get_dataset
from VAE import VAE
import time
import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_EPOCHS = 20
TEST_INTERVAL = 5
N_LATENT = 100
LR = 3e-4


def train(model, train_iter, test_iter, optimizer, num_epochs, test_interval):
    logger.info('Training begins.')
    all_loss = []
    for epoch in range(1, num_epochs+1):
        model.train()
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
        logger.info('Epoch {}, Loss: {:.4f}, ELBO: {:.4f}, Time: {:.4f}'.format(
            epoch, epoch_loss, -epoch_loss, time.time()-start))

        if epoch % test_interval == 0:
            test(model, test_iter, epoch)

    return all_loss


def test(model, test_iter, epoch):
    model.eval()
    start = time.time()

    sum_loss = 0.0
    sum_logpx = 0.0
    n_examples = 0
    with torch.no_grad():
        for batch, X in enumerate(test_iter):
            X = X[0].to(dev)
            y, mu, logvar = model(X)
            loss = model.loss_compute(X, y, mu, logvar)
            logpx = evaluate_LLE(model, X, K=200)

            sum_loss += loss.item() * X[0].shape[0]
            sum_logpx += logpx.item().sum()
            n_examples += X[0].shape[0]
        avg_loss = sum_loss / n_examples
        avg_logpx = sum_logpx / n_examples
    logger.info('Test, Avg Loss: {:.4f}, Avg ELBO: {:.4f}, Avg Log(px): {:.4f}, Time: {:.4f}'.format(
        avg_loss, -avg_loss, avg_logpx, time.time()-start))

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
    plt.savefig('P2_VAE_Epoch_{}_test_samples.png'.format(epoch))


def evaluate_LLE(model, one_batch, K=200):
    def gaussian_distribution(sample, mean, logvar, dim=1):
        log_p_sample = torch.sum(
            -.5 * (logvar + np.log(2*np.pi) + (sample - mean) ** 2. * torch.exp(-logvar)),
            dim=dim)

        return torch.exp(log_p_sample)

    def log_sum_exp(p_xz_raw):
        if len(p_xz_raw.shape) > 2:
            p_xz_raw = p_xz_raw.view(-1, 784)

        log_sum = torch.sum(torch.log(p_xz_raw), dim=1)
        p_xz = torch.exp(log_sum)

        return p_xz

    model.eval()
    # construct prob variables
    p_xz = []
    p_z = []
    q_zx = []
    for k in range(K):
        with torch.no_grad():
            mean_k, logvar_k = model.encode(one_batch)
            z_k = model.reparam(mean_k, logvar_k)
            p_xz_k, logits = model.decode(z_k)
            p_xz_k = log_sum_exp(p_xz_k)
            p_zk = gaussian_distribution(z_k, torch.zeros_like(mean_k), torch.zeros_like(logvar_k))
            q_zx_k = gaussian_distribution(z_k, mean_k, logvar_k)

        p_xz.append(p_xz_k)
        p_z.append(p_zk)
        q_zx.append(q_zx_k)

    all_p_xz = torch.stack(p_xz, dim=1)
    all_p_z = torch.stack(p_z, dim=1)
    all_q_zx = torch.stack(q_zx, dim=1)
    assert all_p_xz.shape == all_p_z.shape == all_q_zx.shape

    inside_log = torch.sum((all_p_xz * all_p_z) / all_q_zx, dim=1) / K
    log_px = torch.log(inside_log)

    return log_px.squeeze()


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