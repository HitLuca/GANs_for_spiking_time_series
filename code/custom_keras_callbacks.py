from math import ceil, sqrt
import keras
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import json


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


class AEModelSaver(keras.callbacks.Callback):
    def __init__(self, autoencoder, encoder, decoder, run_folder):
        super().__init__()
        self._autoencoder = autoencoder
        self._encoder = encoder
        self._decoder = decoder
        self._run_folder = run_folder

    def on_epoch_end(self, epoch, logs={}):
        self._autoencoder.save(os.path.join(self._run_folder, 'autoencoder.h5'))
        self._encoder.save(os.path.join(self._run_folder, 'encoder.h5'))
        self._decoder.save(os.path.join(self._run_folder, 'decoder.h5'))

        config_filepath = os.path.join(self._run_folder, 'config.json')
        config = json.load(open(config_filepath, 'r+'))
        config['epoch'] = epoch
        json.dump(config, open(config_filepath, 'w'))


class AEEmbeddingsVisualizer(keras.callbacks.Callback):
    def __init__(self, encoder, dataset, run_folder):
        super().__init__()
        self._encoder = encoder
        self._run_folder = run_folder
        self._dataset = dataset
        self._dim = int(ceil(sqrt(dataset.shape[1])))
        self._embedded_data = None

    def on_epoch_end(self, epoch, logs={}):
        self._embedded_data = self._encoder.predict(self._dataset, batch_size=32)
        self._visualize_embeddings(epoch)

    def _visualize_embeddings(self, epoch):
        embedding_var = tf.Variable(self._embedded_data, name='embedding')

        embedding_filepath = os.path.join(self._run_folder, 'projector_config.pbtxt')

        with open(embedding_filepath, 'w') as f:
            for i in range(epoch + 1):
                print('embeddings {', file=f)
                print('\ttensor_name: "embedding_' + str(i + 1) + ':0"', file=f)
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
