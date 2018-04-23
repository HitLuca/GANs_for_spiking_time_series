from GAN_utils import *
from keras import Model
from keras.layers import *
from keras.models import load_model
from keras.optimizers import RMSprop
import json

class WGAN:
    def __init__(self, timesteps, latent_dim, run_dir, img_dir, model_dir, generated_datesets_dir):
        self._timesteps = timesteps
        self._latent_dim = latent_dim
        self._run_dir = run_dir
        self._img_dir = img_dir
        self._model_dir = model_dir
        self._generated_datesets_dir = generated_datesets_dir

        self._epoch = 0
        self._losses = [[], []]

        self._save_configuration()

    def build_models(self, generator_lr, critic_lr):
        self._generator = self._build_generator()

        self._critic = self._build_critic()
        self._critic.compile(loss=self._wasserstein_loss, optimizer=RMSprop(critic_lr))

        z = Input(shape=(self._latent_dim,))
        fake = self._generator(z)

        set_model_trainable(self._critic, False)

        valid = self._critic(fake)

        self._gan = Model(z, valid, 'GAN')

        self._gan.compile(
            loss=self._wasserstein_loss,
            optimizer=RMSprop(generator_lr),
            metrics=['accuracy'])

        return self._gan, self._generator, self._critic

    def _build_generator(self):
        generator_inputs = Input((self._latent_dim,))
        generated = generator_inputs

        generated = Lambda(lambda x: K.expand_dims(x))(generated)
        while generated.shape[1] < self._timesteps:
            generated = Conv1D(
                32, 3, activation='relu', padding='same')(generated)
            generated = UpSampling1D(2)(generated)
        generated = Conv1D(
            1, 3, activation='relu', padding='same')(generated)
        generated = Lambda(lambda x: K.squeeze(x, -1))(generated)
        generated = Dense(self._timesteps, activation='tanh')(generated)

        generator = Model(generator_inputs, generated, 'generator')
        return generator

    def _build_critic(self):
        critic_inputs = Input((self._timesteps,))
        criticized = critic_inputs

        #         mbd = MinibatchDiscrimination(5, 3)(criticized)
        #         mbd = Dense(1, activation='tanh')(mbd)

        criticized = Lambda(lambda x: K.expand_dims(x))(
            criticized)
        while criticized.shape[1] > 1:
            criticized = Conv1D(
                32, 3, activation='relu', padding='same')(criticized)
            criticized = MaxPooling1D(2, padding='same')(criticized)
        criticized = Flatten()(criticized)
        criticized = Dense(32, activation='relu')(criticized)
        criticized = Dense(15, activation='relu')(criticized)
        #         criticized = Dense(1, activation='tanh')(criticized)
        #         criticized = Concatenate()([criticized, mbd])
        criticized = Dense(1)(criticized)

        critic = Model(critic_inputs, criticized, 'critic')
        return critic

    def train(self, batch_size, epochs, n_generator, n_critic, dataset, clip_value,
              img_frequency, model_save_frequency, dataset_generation_frequency, dataset_generation_size):
        half_batch = int(batch_size / 2)

        while self._epoch < epochs:
            self._epoch += 1
            for _ in range(n_critic):
                indexes = np.random.randint(0, dataset.shape[0], half_batch)
                batch_transactions = dataset[indexes]

                noise = np.random.normal(0, 1, (half_batch, self._latent_dim))

                generated_transactions = self._generator.predict(noise)

                critic_loss_real = self._critic.train_on_batch(
                    batch_transactions, -np.ones((half_batch, 1)))
                critic_loss_fake = self._critic.train_on_batch(
                    generated_transactions, np.ones((half_batch, 1)))
                critic_loss = 0.5 * np.add(critic_loss_real,
                                           critic_loss_fake)

                self._clip_weights(clip_value)

            for _ in range(n_generator):
                noise = np.random.normal(0, 1, (batch_size, self._latent_dim))

                generator_loss = self._gan.train_on_batch(
                    noise, -np.ones((batch_size, 1)))[0]

            generator_loss = 1 - generator_loss
            critic_loss = 1 - critic_loss

            self._losses[0].append(generator_loss)
            self._losses[1].append(critic_loss)

            print("%d [C loss: %f] [G loss: %f]" % (self._epoch, critic_loss,
                                                    generator_loss))

            if self._epoch % img_frequency == 0:
                self._save_imgs()
                self._save_latent_space()

            if self._epoch % model_save_frequency == 0:
                self._save_models()

            if self._epoch % dataset_generation_frequency == 0:
                self._generate_dataset(self._epoch, dataset_generation_size)

            if self._epoch % 250 == 0:
                self._save_losses()

        self._generate_dataset(epochs, dataset_generation_size)
        self._save_losses()
        self._save_models()
        self._save_imgs()
        self._save_latent_space()

        return self._losses

    def _save_imgs(self):
        filenames = [str(self._img_dir / ('%05d.png' % self._epoch)), str(self._img_dir / 'last.png')]
        generate_save_images(self._generator, 5, 5, self._latent_dim, filenames)

    def _save_latent_space(self):
        filename = str(self._img_dir / ('latent_space.png'))
        save_latent_space(self._latent_dim, self._generator, filename)

    def _save_losses(self):
        save_losses(self._losses, str(self._img_dir / 'losses.png'))

    def _clip_weights(self, clip_value):
        for l in self._critic.layers:
            #             if 'minibatch_discrimination' not in l.name:
            weights = [np.clip(w, -clip_value, clip_value) for w in l.get_weights()]
            l.set_weights(weights)

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

    def _save_configuration(self):
        config = {
            'timesteps': self._timesteps,
            'latent_dim': self._latent_dim
        }

        with open(self._run_dir / 'config.json', 'w') as f:
            json.dump(config, f)

    @staticmethod
    def _wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    def load_models(self):
        custom_objects = {
            'MinibatchDiscrimination': MinibatchDiscrimination,
            '_wasserstein_loss': self._wasserstein_loss
        }

        self._gan = load_model(self._model_dir / 'wgan.h5', custom_objects=custom_objects)
        self._generator = load_model(self._model_dir / 'generator.h5')
        self._critic = load_model(self._model_dir / 'critic.h5', custom_objects=custom_objects)

        return self._gan, self._generator, self._critic