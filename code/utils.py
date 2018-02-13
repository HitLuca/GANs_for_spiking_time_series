import json
import os
from datetime import datetime

import keras
import numpy as np
from matplotlib import pyplot as plt
from scipy import io
from sklearn.preprocessing import MinMaxScaler


def get_balances(rescale, timesteps, balances_filepath):
    def split_balances(balances, timesteps):
        if balances.shape[1] < timesteps:
            return None
        elif balances.shape[1] == timesteps:
            return balances
        else:
            splitted_balances, remaining_balances = np.hsplit(balances, [timesteps])
            remaining_balances = split_balances(remaining_balances, timesteps)
            return np.vstack([splitted_balances, remaining_balances])

    sparse_balances = io.mmread(balances_filepath)
    balances = sparse_balances.todense()
    if rescale:
        balances = MinMaxScaler(feature_range=(-1, 1)).fit_transform(balances)

    balances = split_balances(balances, timesteps)
    return balances


def create_folders():
    folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs('logs/' + folder_name + '/img/')
    return folder_name


class AEResultPlotter(keras.callbacks.Callback):
    def __init__(self, dataset, model, plots_folder):
        super().__init__()
        self._dataset = dataset
        self._model = model
        self._plots_folder = plots_folder

    def on_epoch_end(self, epoch, logs={}):
        plt.subplots(4, 2, figsize=(10, 6))
        N = self._dataset.shape[0]

        indexes = np.random.choice(N, 8, replace=False)
        for i in range(8):
            plt.subplot(4, 2, i + 1)
            plt.plot(self._dataset[indexes[i]])
            result = self._model.predict(np.expand_dims(self._dataset[indexes[i]], 0))
            plt.plot(result.T)
            plt.xticks([])
            plt.ylim(-1, 1)
            plt.yticks([])
        plt.tight_layout()
        plt.savefig(self._plots_folder + str(epoch) + '.png')
        plt.close()


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


class ModelSaver(keras.callbacks.Callback):
    def __init__(self, model, model_filepath, config_filepath):
        super().__init__()
        self._model = model
        self._model_filepath = model_filepath
        self._config_filepath = config_filepath

    def on_epoch_end(self, epoch, logs={}):
        self._model.save(self._model_filepath)

        config = json.load(open(self._config_filepath, 'r+'))
        config['epoch'] = epoch
        json.dump(config, open(self._config_filepath, 'w'))
