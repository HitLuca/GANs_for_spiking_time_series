import pickle
import sys

import vae_utils
from keras.layers import *

sys.path.append("..")
import utils


class VAE:
    def __init__(self, config):
        self._batch_size = config['batch_size']
        self._epochs = config['epochs']
        self._timesteps = config['timesteps']
        self._lr = config['lr']
        self._latent_dim = config['latent_dim']
        self._img_frequency = config['img_frequency']
        self._loss_frequency = config['loss_frequency']
        self._latent_space_frequency = config['latent_space_frequency']
        self._model_save_frequency = config['model_save_frequency']
        self._dataset_generation_frequency = config['dataset_generation_frequency']
        self._dataset_generation_size = config['dataset_generation_size']
        self._run_dir = config['run_dir']
        self._img_dir = config['img_dir']
        self._model_dir = config['model_dir']
        self._generated_datesets_dir = config['generated_datesets_dir']

        self._lr_decay_factor = config['lr_decay_factor']
        self._lr_decay_steps = config['lr_decay_steps']

        self._epoch = 0
        self._losses = []
        self._build_models()

    def _build_models(self):
        self._encoder = vae_utils.build_encoder(self._latent_dim, self._timesteps)
        self._decoder = vae_utils.build_decoder(self._latent_dim, self._timesteps)
        self._vae_model, self._generator = vae_utils.build_vae_model(self._encoder, self._decoder, self._latent_dim, self._timesteps, self._lr)

    def train(self, dataset):
        while self._epoch < self._epochs:
            self._epoch += 1
            indexes = np.random.randint(0, dataset.shape[0], self._batch_size)
            batch_transactions = dataset[indexes].reshape(self._batch_size, self._timesteps)

            vae_loss = self._vae_model.train_on_batch(batch_transactions, batch_transactions)
            self._losses.append(vae_loss)

            print("%d [VAE loss: %f]" % (self._epoch, vae_loss))

            if self._epoch % self._loss_frequency == 0:
                self._save_losses()

            if self._epoch % self._img_frequency == 0:
                self._save_samples()

            if self._epoch % self._latent_space_frequency == 0:
                self._save_latent_space()

            if self._epoch % self._model_save_frequency == 0:
                self._save_models()

            if self._epoch % self._dataset_generation_frequency == 0:
                self._generate_dataset()

            if self._epoch % self._lr_decay_steps == 0:
                self._apply_lr_decay()

        self._generate_dataset()
        self._save_losses()
        self._save_models()
        self._save_samples()
        self._save_latent_space()

    def _save_samples(self):
        rows, columns = 6, 6
        noise = np.random.normal(0, 1, (rows * columns, self._latent_dim))
        generated_transactions = self._generator.predict(noise)

        filenames = [self._img_dir + ('/%07d.png' % self._epoch), self._img_dir + '/last.png']
        utils.save_samples(generated_transactions, rows, columns, filenames)

    def _save_latent_space(self):
        grid_size = 6

        latent_space_inputs = np.zeros((grid_size * grid_size, self._latent_dim))

        for i, v_i in enumerate(np.linspace(-1.5, 1.5, grid_size, True)):
            for j, v_j in enumerate(np.linspace(-1.5, 1.5, grid_size, True)):
                latent_space_inputs[i * grid_size + j, :2] = [v_i, v_j]

        generated_data = self._generator.predict(latent_space_inputs)

        filenames = [self._img_dir + '/latent_space.png']
        utils.save_latent_space(generated_data, grid_size, filenames)

    def _save_losses(self):
        utils.save_losses_other(self._losses, self._img_dir + '/losses.png', 'vae')

        with open(self._run_dir + '/losses.p', 'wb') as f:
            pickle.dump(self._losses, f)

    def _save_models(self):
        self._encoder.save(self._model_dir + '/encoder.h5')
        self._decoder.save(self._model_dir + '/decoder.h5')
        self._vae_model.save(self._model_dir + '/vae_model.h5')
        self._generator.save(self._model_dir + '/generator.h5')

    def _generate_dataset(self):
        z_samples = np.random.normal(0, 1, (self._dataset_generation_size, self._latent_dim))
        generated_dataset = self._generator.predict(z_samples)
        np.save(self._generated_datesets_dir + ('/%d_generated_data' % self._epoch), generated_dataset)
        np.save(self._generated_datesets_dir + '/last', generated_dataset)

    def get_models(self):
        return self._encoder, self._decoder, self._generator, self._vae_model

    def _apply_lr_decay(self):
        lr_tensor = self._vae_model.optimizer.lr
        lr = K.get_value(lr_tensor)
        K.set_value(lr_tensor, lr * self._lr_decay_factor)
