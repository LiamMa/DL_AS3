from torch.utils.data import TensorDataset, Dataset, DataLoader
import torch
import numpy as np
import requests
import os
CODE_PATH = os.path.dirname(__file__)
BASE_URL = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/{}'
BASE_NAME = 'binarized_mnist_{}'


def download_if_not_exists():
    fs = ['train', 'valid', 'test']
    for f in fs:
        nm = BASE_NAME.format(f+'.amat')
        if not os.path.exists(os.path.join(CODE_PATH, 'data', nm)):
            url = BASE_URL.format(nm)
            response = requests.get(url, stream=True)

            with open(os.path.join(CODE_PATH, 'data', nm), 'wb') as writer:
                for data in response.iter_content():
                    writer.write(data)
    print('All .amat files exist.')


def convert_to_np_if_not_exists():
    fs = ['train', 'valid', 'test']
    for f in fs:
        raw_file = os.path.join(CODE_PATH, 'data', BASE_NAME.format(f+'.amat'))
        np_file = os.path.join(CODE_PATH, 'data', BASE_NAME.format(f+'.npy'))
        if not os.path.exists(np_file):
            data = []
            with open(raw_file) as reader:
                for line in reader.readlines():
                    l = [int(x) for x in line[:-1].split(' ')]
                    assert len(l) == 784
                    data.append(l)
            data_np = np.array(data, dtype=np.float32)
            data_np = data_np.reshape((-1, 1, 28, 28))
            np.save(np_file, data_np)
    print('All .npy files exist.')


def get_dataset(f='train'):
    download_if_not_exists()
    convert_to_np_if_not_exists()

    nm = BASE_NAME.format(f+'.npy')
    np_file = os.path.join(CODE_PATH, 'data', nm)
    data_np = np.load(np_file)
    dataset = TensorDataset(torch.from_numpy(data_np))

    return dataset


if __name__ == '__main__':
    train_dataset = get_dataset('train')
    valid_dataset = get_dataset('valid')
    test_dataset = get_dataset('test')