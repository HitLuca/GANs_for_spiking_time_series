import os
import pickle

from keras.layers import *

from generative_models import utils
from generative_models.wgan_gp import wgan_gp_utils


class WGAN_GP:
    def __init__(self, config):
        self._batch_size = config['batch_size']
        self._epochs = config['epochs']
        self._timesteps = config['timesteps']
        self._n_critic = config['n_critic']
        self._n_generator = config['n_generator']
        self._latent_dim = config['latent_dim']

        self._generator_lr = config['generator_lr']
        self._critic_lr = config['critic_lr']
        self._img_frequency = config['img_frequency']
        self._loss_frequency = config['loss_frequency']
        self._latent_space_frequency = config['latent_space_frequency']
        self._model_save_frequency = config['model_save_frequency']
        self._dataset_generation_frequency = config['dataset_generation_frequency']
        self._dataset_generation_size = config['dataset_generation_size']
        self._gradient_penality_weight = config['gradient_penality_weight']
        self._packing_degree = config['packing_degree']
        self._run_dir = config['run_dir']
        self._img_dir = config['img_dir']
        self._model_dir = config['model_dir']
        self._generated_datesets_dir = config['generated_datesets_dir']
        self._use_mbd = config['use_mbd']
        self._use_packing = config['use_packing']

        self._lr_decay_factor = config['lr_decay_factor']
        self._lr_decay_steps = config['lr_decay_steps']

        self._epoch = 0
        self._losses = [[], []]
        self._build_models()

    def _build_models(self):
        self._generator = wgan_gp_utils.build_generator(self._latent_dim, self._timesteps)
        self._critic = wgan_gp_utils.build_critic(self._timesteps, self._use_mbd, self._use_packing,
                                                  self._packing_degree)
        self._generator_model = wgan_gp_utils.build_generator_model(self._generator, self._critic, self._latent_dim,
                                                                    self._timesteps, self._use_packing,
                                                                    self._packing_degree, self._batch_size,
                                                                    self._generator_lr)
        self._critic_model = wgan_gp_utils.build_critic_model(self._generator, self._critic, self._latent_dim,
                                                              self._timesteps, self._use_packing, self._packing_degree,
                                                              self._batch_size,
                                                              self._critic_lr, self._gradient_penality_weight)

    def train(self, dataset):
        ones = np.ones((self._batch_size, 1))
        neg_ones = -ones
        zeros = np.zeros((self._batch_size, 1))

        while self._epoch < self._epochs:
            self._epoch += 1
            critic_losses = []
            for _ in range(self._n_critic):
                indexes = np.random.randint(0, dataset.shape[0], self._batch_size)
                batch_transactions = dataset[indexes].reshape(self._batch_size, self._timesteps)
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = [batch_transactions, noise]

                if self._use_packing:
                    supporting_indexes = np.random.randint(0, dataset.shape[0],
                                                           (self._batch_size * self._packing_degree))
                    supporting_transactions = dataset[supporting_indexes].reshape(self._batch_size, self._timesteps,
                                                                                  self._packing_degree)
                    supporting_noise = np.random.normal(0, 1,
                                                        (self._batch_size, self._latent_dim, self._packing_degree))
                    inputs.extend([supporting_transactions, supporting_noise])

                critic_losses.append(self._critic_model.train_on_batch(inputs, [ones, neg_ones, zeros])[0])
            critic_loss = np.mean(critic_losses)

            generator_losses = []
            for _ in range(self._n_generator):
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = [noise]

                if self._use_packing:
                    supporting_noise = np.random.normal(0, 1,
                                                        (self._batch_size, self._latent_dim, self._packing_degree))
                    inputs.append(supporting_noise)

                generator_losses.append(self._generator_model.train_on_batch(inputs, ones))
            generator_loss = np.mean(generator_losses)

            generator_loss = float(-generator_loss)
            critic_loss = float(-critic_loss)

            self._losses[0].append(generator_loss)
            self._losses[1].append(critic_loss)

            print("%d [C loss: %+.6f] [G loss: %+.6f]" % (self._epoch, critic_loss, generator_loss))

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

        return self._losses

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
        utils.save_losses_wgan(self._losses, self._img_dir + '/losses.png')

        with open(self._run_dir + '/losses.p', 'wb') as f:
            pickle.dump(self._losses, f)

    def _save_models(self):
        dir = self._model_dir + '/' + str(self._epoch) + '/'
        os.mkdir(dir)
        self._generator_model.save(dir + 'generator_model.h5')
        self._critic_model.save(dir + 'critic_model.h5')
        self._generator.save(dir + 'generator.h5')
        self._critic.save(dir + 'critic.h5')

    def _generate_dataset(self):
        z_samples = np.random.normal(0, 1, (self._dataset_generation_size, self._latent_dim))
        generated_dataset = self._generator.predict(z_samples)
        np.save(self._generated_datesets_dir + ('/%d_generated_data' % self._epoch), generated_dataset)
        np.save(self._generated_datesets_dir + '/last', generated_dataset)

    def get_models(self):
        return self._generator, self._critic, self._generator_model, self._critic_model

    def _apply_lr_decay(self):
        lr_tensor = self._generator_model.optimizer.lr
        lr = K.get_value(lr_tensor)
        K.set_value(lr_tensor, lr * self._lr_decay_factor)

        lr_tensor = self._critic_model.optimizer.lr
        lr = K.get_value(lr_tensor)
        K.set_value(lr_tensor, lr * self._lr_decay_factor)
