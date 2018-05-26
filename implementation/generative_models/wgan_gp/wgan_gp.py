import pickle
from functools import partial

from implementation.generative_models import utils
from keras import Model
from keras.layers import *
from keras.optimizers import *


class WGAN_GP:
    def __init__(self, timesteps, latent_dim, packing_degree, gradient_penality_weight, batch_size, run_dir, img_dir,
                 model_dir, generated_datesets_dir):
        self._timesteps = timesteps
        self._latent_dim = latent_dim
        self._run_dir = run_dir
        self._img_dir = img_dir
        self._model_dir = model_dir
        self._generated_datesets_dir = generated_datesets_dir

        self._save_config()

        self._epoch = 0
        self._losses = [[], []]

        self._batch_size = batch_size
        self._packing_degree = packing_degree

        self._gradient_penality_weight = gradient_penality_weight

    def build_models(self, generator_lr, critic_lr):
        self._generator = self._build_generator()
        self._critic = self._build_critic()
        self._generator_model = self._build_generator_model(generator_lr)
        self._critic_model = self._build_critic_model(critic_lr)

        return self._generator, self._critic

    def _build_generator_model(self, generator_lr):
        utils.set_model_trainable(self._generator, True)
        utils.set_model_trainable(self._critic, False)

        noise_samples = Input((self._latent_dim,))
        supporting_noise_samples = Input((self._latent_dim, self._packing_degree))

        reshaped_supporting_noise_samples = Lambda(lambda x: K.reshape(x, (self._batch_size * self._packing_degree, self._latent_dim)))(supporting_noise_samples)
        generated_samples = self._generator(noise_samples)
        generated_samples = Lambda(lambda x: K.reshape(x, (self._batch_size, self._timesteps, 1)))(generated_samples)

        supporting_generated_samples = self._generator(reshaped_supporting_noise_samples)
        supporting_generated_samples = Lambda(
            lambda x: K.reshape(x, (self._batch_size, self._timesteps, self._packing_degree)))(
            supporting_generated_samples)
        merged_generated_samples = Concatenate(-1)([generated_samples, supporting_generated_samples])

        generated_criticized = self._critic(merged_generated_samples)

        generator_model = Model([noise_samples, supporting_noise_samples], generated_criticized, 'generator_model')
        generator_model.compile(optimizer=Adam(generator_lr, beta_1=0.5, beta_2=0.9), loss=utils.wasserstein_loss)
        return generator_model

    def _build_critic_model(self, critic_lr):
        utils.set_model_trainable(self._generator, False)
        utils.set_model_trainable(self._critic, True)

        noise_samples = Input((self._latent_dim,))
        real_samples = Input((self._timesteps,))
        supporting_noise_samples = Input((self._latent_dim, self._packing_degree))
        supporting_real_samples = Input((self._timesteps, self._packing_degree))

        reshaped_supporting_noise_samples = Lambda(lambda x: K.reshape(x, (self._batch_size * self._packing_degree, self._latent_dim)))(supporting_noise_samples)
        generated_samples = self._generator(noise_samples)
        supporting_generated_samples = self._generator(reshaped_supporting_noise_samples)

        expanded_generated_samples = Lambda(lambda x: K.reshape(x, (self._batch_size, self._timesteps, 1)))(
            generated_samples)
        expanded_generated_supporting_samples = Lambda(
            lambda x: K.reshape(x, (self._batch_size, self._timesteps, self._packing_degree)))(
            supporting_generated_samples)

        merged_generated_samples = Concatenate(-1)([expanded_generated_samples, expanded_generated_supporting_samples])

        generated_criticized = self._critic(merged_generated_samples)

        expanded_real_samples = Lambda(lambda x: K.reshape(x, (self._batch_size, self._timesteps, 1)))(real_samples)
        merged_real_samples = Lambda(lambda x: K.concatenate(x, -1))([expanded_real_samples, supporting_real_samples])

        real_criticized = self._critic(merged_real_samples)

        averaged_samples = utils.RandomWeightedAverage(self._batch_size)([real_samples, generated_samples])

        expanded_averaged_samples = Lambda(lambda x: K.reshape(x, (self._batch_size, self._timesteps, 1)))(
            averaged_samples)

        expanded_supporting_real_samples = Lambda(
            lambda x: K.reshape(x, ((self._batch_size * self._packing_degree), self._timesteps)))(
            supporting_real_samples)
        averaged_support_samples = utils.RandomWeightedAverage((self._batch_size * self._packing_degree))(
            [expanded_supporting_real_samples, supporting_generated_samples])

        averaged_support_samples = Lambda(
            lambda x: K.reshape(x, (self._batch_size, self._timesteps, self._packing_degree)))(
            averaged_support_samples)

        merged_averaged_samples = Concatenate(-1)([expanded_averaged_samples, averaged_support_samples])

        averaged_criticized = self._critic(merged_averaged_samples)

        partial_gp_loss = partial(utils.gradient_penalty_loss,
                                  averaged_samples=merged_averaged_samples,
                                  gradient_penalty_weight=self._gradient_penality_weight)
        partial_gp_loss.__name__ = 'gradient_penalty'

        critic_model = Model([real_samples, supporting_real_samples, noise_samples, supporting_noise_samples],
                             [real_criticized, generated_criticized, averaged_criticized], 'critic_model')

        critic_model.compile(optimizer=Adam(critic_lr, beta_1=0.5, beta_2=0.9),
                             loss=[utils.wasserstein_loss, utils.wasserstein_loss, partial_gp_loss])
        return critic_model

    def _build_generator(self):
        generator_inputs = Input((self._latent_dim,))
        generated = generator_inputs

        if self._latent_dim != 15:
            generated = Dense(15)(generated)
            generated = BatchNormalization()(generated)
            generated = LeakyReLU(0.2)(generated)

        generated = Lambda(lambda x: K.expand_dims(x))(generated)

        generated = Conv1D(32, 2, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        generated = Conv1D(32, 2, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        generated = UpSampling1D(2)(generated)

        generated = Conv1D(32, 2, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        generated = Conv1D(32, 2, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        generated = UpSampling1D(2)(generated)

        generated = Conv1D(32, 2, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        generated = Conv1D(32, 2, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        generated = UpSampling1D(2)(generated)

        generated = Conv1D(1, 1, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)

        generated = Lambda(lambda x: K.squeeze(x, -1))(generated)

        generated = Dense(self._timesteps)(generated)
        generated = BatchNormalization()(generated)
        generated = Activation('tanh')(generated)

        generator = Model(generator_inputs, generated, 'generator')
        return generator

    def _build_critic(self):
        critic_inputs = Input((self._timesteps, self._packing_degree + 1))
        criticized = critic_inputs

        criticized = Conv1D(32, 2)(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        # criticized = Conv1D(32, 2)(criticized)
        # criticized = LeakyReLU(0.2)(criticized)
        criticized = MaxPooling1D(2, padding='same')(criticized)

        criticized = Conv1D(32, 2)(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        # criticized = Conv1D(32, 2)(criticized)
        # criticized = LeakyReLU(0.2)(criticized)
        criticized = MaxPooling1D(2, padding='same')(criticized)

        criticized = Conv1D(32, 2)(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        # criticized = Conv1D(32, 2)(criticized)
        # criticized = LeakyReLU(0.2)(criticized)
        criticized = MaxPooling1D(2, padding='same')(criticized)

        criticized = Flatten()(criticized)

        criticized = Dense(15)(criticized)
        criticized = LeakyReLU(0.2)(criticized)

        criticized = Dense(1)(criticized)

        critic = Model(critic_inputs, criticized, 'critic')

        return critic

    def train(self, epochs, n_generator, n_critic, dataset, img_frequency, loss_frequency, latent_space_frequency,
              model_save_frequency, dataset_generation_frequency, dataset_generation_size):

        ones = np.ones((self._batch_size, 1))
        neg_ones = -ones
        zeros = np.zeros((self._batch_size, 1))

        while self._epoch < epochs:
            self._epoch += 1
            critic_losses = []
            for _ in range(n_critic):
                indexes = np.random.randint(0, dataset.shape[0], (self._batch_size, self._packing_degree + 1))
                batch_transactions = dataset[indexes[:, 0]].reshape(self._batch_size, self._timesteps)
                supporting_transactions = dataset[indexes[:, 1:]].reshape(self._batch_size, self._timesteps,
                                                                          self._packing_degree)

                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                supporting_noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim, self._packing_degree))

                critic_losses.append(
                    self._critic_model.train_on_batch(
                        [batch_transactions, supporting_transactions, noise, supporting_noise],
                        [ones, neg_ones, zeros]))
            critic_loss = np.mean(critic_losses)

            generator_losses = []
            for _ in range(n_generator):
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                supporting_noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim, self._packing_degree))

                generator_losses.append(
                    self._generator_model.train_on_batch([noise, supporting_noise], ones))
            generator_loss = np.mean(generator_losses)

            generator_loss = float(-generator_loss)
            critic_loss = float(-critic_loss)

            self._losses[0].append(generator_loss)
            self._losses[1].append(critic_loss)

            print("%d [C loss: %f] [G loss: %f]" % (self._epoch, critic_loss, generator_loss))

            if self._epoch % loss_frequency == 0:
                self._save_losses()

            if self._epoch % img_frequency == 0:
                self._save_samples()

            if self._epoch % latent_space_frequency == 0:
                self._save_latent_space()

            if self._epoch % model_save_frequency == 0:
                self._save_models()

            if self._epoch % dataset_generation_frequency == 0:
                self._generate_dataset(self._epoch, dataset_generation_size)

        self._generate_dataset(epochs, dataset_generation_size)
        self._save_losses()
        self._save_models()
        self._save_samples()
        self._save_latent_space()

        return self._losses

    def _save_samples(self):
        rows, columns = 6, 6
        noise = np.random.normal(0, 1, (rows * columns, self._latent_dim))
        generated_transactions = self._generator.predict(noise)

        filenames = [str(self._img_dir / ('%07d.png' % self._epoch)), str(self._img_dir / 'last.png')]
        utils.save_samples(generated_transactions, rows, columns, filenames, False)

    def _save_latent_space(self):
        grid_size = 6

        latent_space_inputs = np.zeros((grid_size * grid_size, self._latent_dim))

        for i, v_i in enumerate(np.linspace(-1.5, 1.5, grid_size, True)):
            for j, v_j in enumerate(np.linspace(-1.5, 1.5, grid_size, True)):
                latent_space_inputs[i * grid_size + j, :2] = [v_i, v_j]

        generated_data = self._generator.predict(latent_space_inputs)

        filenames = [str(self._img_dir / 'latent_space.png')]
        utils.save_latent_space(generated_data, grid_size, filenames, False)

    def _save_losses(self):
        utils.save_losses(self._losses, str(self._img_dir / 'losses.png'))

        with open(str(self._run_dir / 'losses.p'), 'wb') as f:
            pickle.dump(self._losses, f)

    def _save_config(self):
        config = {
            'timesteps': self._timesteps,
            'latent_dim': self._latent_dim,
            'run_dir': self._run_dir,
            'img_dir': self._img_dir,
            'model_dir': self._model_dir,
            'generated_datesets_dir': self._generated_datesets_dir
        }

        with open(str(self._run_dir / 'config.p'), 'wb') as f:
            pickle.dump(config, f)

    def _save_models(self):
        # self._generator_model.save(self._model_dir / 'generator_model.h5')
        # self._critic_model.save(self._model_dir / 'critic_model.h5')
        self._generator.save(self._model_dir / 'generator.h5')
        # self._critic.save(self._model_dir / 'critic.h5')

    def _generate_dataset(self, epoch, dataset_generation_size):
        z_samples = np.random.normal(0, 1, (dataset_generation_size, self._latent_dim))
        generated_dataset = self._generator.predict(z_samples)
        np.save(self._generated_datesets_dir / ('%d_generated_data' % epoch), generated_dataset)
        np.save(self._generated_datesets_dir / 'last', generated_dataset)

    def get_models(self):
        return self._generator, self._critic, self._generator_model, self._critic_model
