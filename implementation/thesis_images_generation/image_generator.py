from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context("paper")

import sys
sys.path.append("..")
from generative_models import utils
import os
import pickle


class ImageGenerator:
    def __init__(self, filepath, latent_dim=2, losses_end=-1):
        self._filepath = filepath
        self._latent_dim = latent_dim
        self._losses_end = losses_end
        if os.path.splitext(self._filepath)[1] == '.h5':
            self._generator_model = load_model(self._filepath,
                                               custom_objects={'BatchNormalizationGAN': utils.BatchNormalizationGAN,
                                                               'wasserstein_loss': utils.wasserstein_loss})
        elif os.path.splitext(self._filepath)[1] == '.p':
            self._losses = pickle.load(open(self._filepath, 'rb'))

    def generate_samples(self, rows=6, columns=6):
        plt.subplots(rows, columns, figsize=(2 * columns, 1 * rows))

        noise = np.random.normal(0, 1, (rows * columns, self._latent_dim))
        generated_data = self._generator_model.predict(noise)

        for row in range(rows):
            for column in range(columns):
                index = row * rows + column
                plt.subplot(rows, columns, index + 1)
                # sns.set_style("white")
                plt.plot(generated_data[index])
                # sns.set_style({"axes.linewidth": "2"})

                plt.ylim([-1, 1])
                plt.xticks([])
                plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

    def generate_latent_space(self, grid_size=6, latent_space_limit=1.5):
        plt.subplots(grid_size, grid_size, figsize=(2 * grid_size, 1 * grid_size))

        latent_space_inputs = np.zeros((grid_size * grid_size, self._latent_dim))

        for i, v_i in enumerate(np.linspace(-latent_space_limit, latent_space_limit, grid_size, True)):
            for j, v_j in enumerate(np.linspace(-latent_space_limit, latent_space_limit, grid_size, True)):
                latent_space_inputs[i * grid_size + j, :2] = [v_i, v_j]

        generated_data = self._generator_model.predict(latent_space_inputs)

        for row in range(grid_size):
            for column in range(grid_size):
                index = row * grid_size + column
                plt.subplot(grid_size, grid_size, index + 1)
                plt.plot(generated_data[index])
                plt.ylim([-1, 1])
                plt.xticks([])
                plt.yticks([])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.show()

    def generate_losses(self, mean_filter=False, mean_filter_size=100):
        print(len(self._losses[0]))
        if mean_filter:
            self._losses[0] = np.convolve(self._losses[0], np.ones((mean_filter_size, )) / mean_filter_size, mode='valid')
            self._losses[1] = np.convolve(self._losses[1], np.ones((mean_filter_size, )) / mean_filter_size, mode='valid')

        plt.figure(figsize=(15, 9))
        plt.plot(self._losses[0][:self._losses_end])
        plt.plot(self._losses[1][:self._losses_end])
        plt.legend(['generator', 'critic'])
        plt.show()
