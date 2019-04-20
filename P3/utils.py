import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import os
import time
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = 'ckpt'
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def model_save(model, save_name):
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    origin = next(iter(model.parameters())).device
    model = model.cpu()
    save_path = os.path.join(CHECKPOINT_PATH, save_name)
    torch.save(model, save_path)
    logger.info('Save model to <{}>'.format(save_path))
    model.to(origin)


def model_load(load_name):
    load_path = os.path.join(CHECKPOINT_PATH, load_name)
    print(load_path)
    model = torch.load(load_path)
    model.to(dev)
    logger.info('Load model <{}> to <{}>'.format(load_path, dev))
    return model


def make_grid_img(all_imgs, save_name, title=None):
    """
    grid_img should be a 5 dim np.ndarray: [grid_i, grid_j, channel, width, height]
    """
    if all_imgs.min() < 0:
        all_imgs = de_normalize(all_imgs)

    save_path = 'figure'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, save_name)
    num_i, num_j = all_imgs.shape[0], all_imgs.shape[1]
    fig, axes = plt.subplots(num_i, num_j, figsize=(num_j*2, num_i*2))
    plt.tight_layout()
    for i in range(all_imgs.shape[0]):
        for j in range(all_imgs.shape[1]):
            axes[i, j].imshow(all_imgs[i, j])
            axes[i, j].axis('off')
    if title is not None:
        fig.suptitle(title)
    plt.savefig(save_path)
    plt.close()
    logger.info('Save image (size: {}) on: {}'.format(all_imgs.shape, save_path))


def de_normalize(all_imgs):
    """
    de-normalize images from [-1, 1] to [0, 1]
    :param all_imgs: img matrix
    :return:
    """
    return (all_imgs + 1.0) / 2.0


def make_plot(y, x=None, save_name='example', title=None, tickets=['Epochs', 'Loss']):
    if x is None:
        x = np.arange(len(y))
    plt.plot(x, y)
    plt.xlabel(tickets[0])
    plt.ylabel(tickets[1])
    if title:
        plt.title(title)
    save_path = 'figure'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, save_name)
    plt.savefig(save_path)

