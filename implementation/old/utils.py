import json
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import ceil, sqrt


def get_balances(balances_filepath, timesteps=100, rescale=True):
    def split_balances(balances, timesteps):
        if balances.shape[1] < timesteps:
            return None
        elif balances.shape[1] == timesteps:
            return balances
        else:
            splitted_balances, remaining_balances = np.hsplit(balances, [timesteps])
            remaining_balances = split_balances(remaining_balances, timesteps)
            if remaining_balances is not None:
                return np.vstack([splitted_balances, remaining_balances])
            return splitted_balances

    balances = np.load(balances_filepath)
    if rescale:
        balances = MinMaxScaler(feature_range=(-1, 1)).fit_transform(balances)
    balances = split_balances(balances, timesteps)
    return balances


def get_date_time():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


class AAEResultPlotter:
    def __init__(self, dataset, encoder, decoder, plots_folder):
        self._dataset = dataset
        self._encoder = encoder
        self._decoder = decoder
        self._plots_folder = plots_folder

    def plot_results(self, epoch):
        plt.subplots(4, 2, figsize=(10, 6))
        N = self._dataset.shape[0]

        indexes = np.random.choice(N, 8, replace=False)
        for i in range(8):
            plt.subplot(4, 2, i + 1)
            plt.plot(self._dataset[indexes[i]])
            result = self._decoder.predict(self._encoder.predict(np.expand_dims(self._dataset[indexes[i]], 0)))
            plt.plot(result.T)
            plt.xticks([])
            plt.ylim(-1, 1)
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(self._plots_folder + str(epoch) + '.png')
        plt.close()


def create_dataset_sprite(dataset, run_folder):
    N, D = dataset.shape

    size = int(ceil(sqrt(N)))
    dim = int(ceil(sqrt(D)))
    padded = np.zeros((N, dim * dim))
    padded[:, :D] = dataset

    sprite_image = np.zeros((size * dim, size * dim))

    k = 0
    for i in range(0, size * dim, dim):
        for j in range(0, size * dim, dim):
            if k < N:
                sprite_image[i:i + dim, j:j + dim] = np.reshape(padded[k], newshape=(dim, dim))
                k += 1

    plt.imsave(os.path.join(run_folder, 'sprite.png'), sprite_image, cmap='plasma')


def save_config(timesteps, batch_num, lstm_size, latent_dim, lr, epoch, epochs, print_freq, config_filepath):
    config = {'batch_num': batch_num,
              'timesteps': timesteps,
              'lstm_size': lstm_size,
              'latent_dim': latent_dim,
              'lr': lr,
              'epochs': epochs,
              'epoch': epoch,
              'print_freq': print_freq}

    json.dump(config, open(config_filepath, 'w'))


def load_config(config_filepath):
    config = json.load(open(config_filepath, 'r'))

    batch_num = config['batch_num']
    timesteps = config['timesteps']
    epochs = config['epochs']
    epoch = config['epoch'] + 1
    lstm_size = config['lstm_size']
    latent_dim = config['latent_dim']
    lr = config['lr']
    print_freq = config['print_freq']

    return timesteps, batch_num, lstm_size, latent_dim, lr, epoch, epochs, print_freq
