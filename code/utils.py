import json
import os
from datetime import datetime

import keras
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
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
        fig_filepath = os.path.join(self._plots_folder, str(epoch) + '.png')
        plt.savefig(fig_filepath)
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
    def __init__(self, model, run_folder):
        super().__init__()
        self._model = model
        self._run_folder = run_folder

    def on_epoch_end(self, epoch, logs={}):
        self._model.save(os.path.join(self._run_folder, 'model.h5'))

        config_filepath = os.path.join(self._run_folder, 'config.json')
        config = json.load(open(config_filepath, 'r+'))
        config['epoch'] = epoch
        json.dump(config, open(config_filepath, 'w'))


def create_dataset_sprite(dataset, run_folder):
    N, D = dataset.shape

    size = int(ceil(sqrt(N)))
    dim = int(ceil(sqrt(D)))
    padded = np.zeros((N, dim*dim))
    padded[:, :D] = dataset

    sprite_image = np.zeros((size * dim, size * dim))

    k = 0
    for i in range(0, size * dim, dim):
        for j in range(0, size * dim, dim):
            if k < N:
                sprite_image[i:i + dim, j:j + dim] = np.reshape(padded[k], newshape=(dim, dim))
                k += 1

    plt.imsave(os.path.join(run_folder, 'sprite.png'), sprite_image, cmap='plasma')


class EmbeddingsVisualizer(keras.callbacks.Callback):
    def __init__(self, encoder, dataset, run_folder):
        super().__init__()
        self._encoder = encoder
        self._run_folder = run_folder
        self._dataset = dataset
        self._dim = int(ceil(sqrt(dataset.shape[1])))
        self._embedded_data = None

    def on_epoch_end(self, epoch, logs={}):
        self._embedded_data = self._encoder.predict(self._dataset)
        self._visualize_embeddings(epoch)

    def _visualize_embeddings(self, epoch):
        summary_writer = tf.summary.FileWriter(self._run_folder)
        embedding_var = tf.Variable(self._embedded_data, name='embedding')

        embedding_filepath = os.path.join(self._run_folder, 'projector_config.pbtxt')

        # config = projector.ProjectorConfig()
        # embedding = config.embeddings.add()
        # embedding.tensor_name = embedding_var.name
        #
        # embedding.sprite.image_path = 'sprite.png'
        # embedding.sprite.single_image_dim.extend([self._dim, self._dim])
        #
        # projector.visualize_embeddings(summary_writer, config)

        with open(embedding_filepath, 'w') as f:
            for i in range(epoch+1):
                print('embeddings {', file=f)
                print('\ttensor_name: "embedding_' + str(i+1) + ':0"', file=f)
                print('\tsprite {', file=f)
                print('\t\timage_path: "sprite.png"', file=f)
                print('\t\tsingle_image_dim: 10', file=f)
                print('\t\tsingle_image_dim: 10', file=f)
                print('\t}', file=f)
                print('}', file=f)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(self._run_folder, 'model.ckpt'))


