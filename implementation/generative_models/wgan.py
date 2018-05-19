import pickle

import utils
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop
from keras.initializers import RandomNormal


class WGAN:
    def __init__(self, timesteps, latent_dim, packing_degree, run_dir, img_dir, model_dir, generated_datesets_dir):
        self._timesteps = timesteps
        self._latent_dim = latent_dim
        self._run_dir = run_dir
        self._img_dir = img_dir
        self._model_dir = model_dir
        self._generated_datesets_dir = generated_datesets_dir

        self._save_config()

        self._epoch = 0
        self._losses = [[], []]

        self._packing_degree = packing_degree
        self._weights_initializer = RandomNormal(stddev=0.02)

    def build_models(self, generator_lr, critic_lr):
        self._generator = self._build_generator()

        self._critic = self._build_critic()
        self._critic.compile(loss=utils.wasserstein_loss, optimizer=RMSprop(critic_lr))

        z = Input((self._latent_dim,))
        supporting_inputs = Input((self._timesteps, self._packing_degree))

        fake = self._generator(z)

        utils.set_model_trainable(self._critic, False)
        # valid = self._critic(fake)
        valid = self._critic([fake, supporting_inputs])

        self._gan = Model([z, supporting_inputs], valid, 'WGAN')
        # self._gan = Model(z, valid, 'GAN')

        self._gan.compile(
            loss=utils.wasserstein_loss,
            optimizer=RMSprop(generator_lr))

        return self._gan, self._generator, self._critic

    def _build_generator(self):
        generator_inputs = Input((self._latent_dim,))
        generated = generator_inputs

        if self._latent_dim != 12:
            generated = Dense(12)(generated)
            generated = BatchNormalization()(generated)
            generated = Activation('relu')(generated)

        generated = Lambda(lambda x: K.expand_dims(x))(generated)

        generated = UpSampling1D(2)(generated)
        generated = Conv1D(64, 3, padding='same', kernel_initializer=self._weights_initializer)(generated)
        generated = BatchNormalization()(generated)
        generated = Activation('relu')(generated)

        generated = UpSampling1D(2)(generated)
        generated = Conv1D(32, 3, padding='same', kernel_initializer=self._weights_initializer)(generated)
        generated = BatchNormalization()(generated)
        generated = Activation('relu')(generated)

        generated = UpSampling1D(2)(generated)
        generated = Conv1D(16, 3, padding='same', kernel_initializer=self._weights_initializer)(generated)
        generated = BatchNormalization()(generated)
        generated = Activation('relu')(generated)
        generated = Conv1D(1, 1, padding='same', kernel_initializer=self._weights_initializer)(generated)
        generated = BatchNormalization()(generated)
        generated = Activation('relu')(generated)

        generated = Lambda(lambda x: K.squeeze(x, -1))(generated)
        generated = Dense(self._timesteps, activation='tanh')(generated)

        generator = Model(generator_inputs, generated, 'generator')
        return generator

    def _build_critic(self):
        critic_inputs = Input((self._timesteps,))
        supporting_inputs = Input((self._timesteps, self._packing_degree))

        criticized = critic_inputs

        criticized = Lambda(lambda x: K.expand_dims(x))(criticized)
        criticized = Concatenate(-1)([criticized, supporting_inputs])

        criticized = Conv1D(16, 3, padding='same', kernel_initializer=self._weights_initializer)(criticized)
        criticized = BatchNormalization()(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        criticized = MaxPooling1D(2, padding='same')(criticized)

        criticized = Conv1D(32, 3, padding='same', kernel_initializer=self._weights_initializer)(criticized)
        criticized = BatchNormalization()(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        criticized = MaxPooling1D(2, padding='same')(criticized)

        criticized = Conv1D(64, 3, padding='same', kernel_initializer=self._weights_initializer)(criticized)
        criticized = BatchNormalization()(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        criticized = MaxPooling1D(2, padding='same')(criticized)

        criticized = Flatten()(criticized)
        criticized = Dense(1)(criticized)

        critic = Model([critic_inputs, supporting_inputs], criticized, 'critic')

        return critic

    def train(self, batch_size, epochs, n_generator, n_critic, dataset, clip_value,
              img_frequency, loss_frequency, latent_space_frequency, model_save_frequency, dataset_generation_frequency,
              dataset_generation_size):
        half_batch = int(batch_size / 2)

        while self._epoch < epochs:
            self._epoch += 1
            critic_losses = []
            for _ in range(n_critic):
                indexes = np.random.randint(0, dataset.shape[0], (half_batch, self._packing_degree + 1))
                batch_transactions = dataset[indexes[:, 0]].reshape(half_batch, self._timesteps)
                supporting_transactions = dataset[indexes[:, 1:]].reshape(half_batch, self._timesteps, self._packing_degree)

                noise = np.random.normal(0, 1, (half_batch * (self._packing_degree + 1), self._latent_dim))
                generated_transactions = self._generator.predict(noise)
                supporting_generated_transactions = generated_transactions[half_batch:].reshape(half_batch, self._timesteps, self._packing_degree)
                generated_transactions = generated_transactions[:half_batch].reshape(half_batch, self._timesteps)

                critic_loss_real = self._critic.train_on_batch([batch_transactions, supporting_transactions], -np.ones((half_batch, 1)))
                critic_loss_fake = self._critic.train_on_batch([generated_transactions, supporting_generated_transactions], np.ones((half_batch, 1)))
                # critic_loss_real = self._critic.train_on_batch(batch_transactions, -np.ones((half_batch, 1)))
                # critic_loss_fake = self._critic.train_on_batch(generated_transactions, np.ones((half_batch, 1)))

                critic_losses.append(0.5 * np.add(critic_loss_real, critic_loss_fake))

                utils.clip_weights(self._critic, clip_value)
            critic_loss = np.mean(critic_losses)

            generator_losses = []
            for _ in range(n_generator):
                noise = np.random.normal(0, 1, (batch_size, self._latent_dim))
                if self._packing_degree > 0:
                    supporting_noise = np.random.normal(0, 1, (batch_size * self._packing_degree, self._latent_dim))
                    supporting_generated_transactions = self._generator.predict(supporting_noise).reshape(batch_size, self._timesteps, self._packing_degree)
                else:
                    supporting_generated_transactions = np.empty((batch_size, self._timesteps, 0))

                generator_losses.append(self._gan.train_on_batch([noise, supporting_generated_transactions], -np.ones((batch_size, 1))))

            generator_loss = np.mean(generator_losses)

            generator_loss = 1 - generator_loss
            critic_loss = 1 - critic_loss

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
        rows, columns = 5, 5
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
        self._gan.save(self._model_dir / 'wgan.h5')
        self._generator.save(self._model_dir / 'generator.h5')
        self._critic.save(self._model_dir / 'critic.h5')

    def _generate_dataset(self, epoch, dataset_generation_size):
        z_samples = np.random.normal(0, 1, (dataset_generation_size, self._latent_dim))
        generated_dataset = self._generator.predict(z_samples)
        np.save(self._generated_datesets_dir / ('%d_generated_data' % epoch), generated_dataset)
        np.save(self._generated_datesets_dir / 'last', generated_dataset)

    def get_models(self):
        return self._gan, self._generator, self._critic
