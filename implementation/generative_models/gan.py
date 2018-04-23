from gan_utils import *
from keras import Model
from keras.layers import *
from keras.optimizers import *


class GAN:
    def __init__(self, timesteps, latent_dim):
        self._timesteps = timesteps
        self._latent_dim = latent_dim

    def build_model(self, lr):
        self._generator = self._build_generator()

        self._discriminator = self._build_discriminator()
        self._discriminator.compile(loss='binary_crossentropy', optimizer=RMSprop(lr))

        z = Input(shape=(self._latent_dim,))
        fake = self._generator(z)

        set_model_trainable(self._discriminator, False)

        valid = self._discriminator(fake)

        self._gan = Model(z, valid, 'GAN')

        self._gan.compile(loss='binary_crossentropy', optimizer=RMSprop(lr))

        return self._gan, self._generator, self._discriminator

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

    def _build_discriminator(self):
        discriminator_inputs = Input((self._timesteps,))
        discriminated = discriminator_inputs

        mbd = MinibatchDiscrimination(5, 3)(discriminated)
        mbd = Dense(1, activation='relu')(mbd)

        discriminated = Lambda(lambda x: K.expand_dims(x))(discriminated)
        while discriminated.shape[1] > 1:
            discriminated = Conv1D(32, 3, activation='relu', padding='same')(discriminated)
            discriminated = MaxPooling1D(2, padding='same')(discriminated)
        discriminated = Flatten()(discriminated)
        #         discriminated = Dense(32, activation='tanh')(discriminated)
        #         discriminated = Dense(15, activation='relu')(discriminated)
        #         discriminated = Dense(1, activation='relu')(discriminated)
        discriminated = Concatenate()([discriminated, mbd])
        discriminated = Dense(1, activation='sigmoid')(discriminated)

        discriminator = Model(discriminator_inputs, discriminated, 'discriminator')
        return discriminator

    def train(self, batch_size, epochs, n_generator, n_discriminator, dataset,
              img_frequency):
        half_batch = int(batch_size / 2)

        losses = [[], []]
        for epoch in range(epochs):
            for _ in range(n_discriminator):
                indexes = np.random.randint(0, dataset.shape[0], half_batch)
                batch_transactions = dataset[indexes]

                noise = np.random.normal(0, 1, (half_batch, self._latent_dim))

                generated_transactions = self._generator.predict(noise)

                discriminator_loss_real = self._discriminator.train_on_batch(
                    batch_transactions, np.random.uniform(0.66, 1, (half_batch, 1)))
                discriminator_loss_fake = self._discriminator.train_on_batch(
                    generated_transactions, np.random.uniform(0, 0.33, (half_batch, 1)))
                discriminator_loss = 0.5 * np.add(discriminator_loss_real,
                                                  discriminator_loss_fake)

            for _ in range(n_generator):
                noise = np.random.normal(0, 1, (batch_size, self._latent_dim))

                generator_loss = self._gan.train_on_batch(
                    noise, np.ones((batch_size, 1)))

            losses[0].append(generator_loss)
            losses[1].append(discriminator_loss)

            print("%d [D loss: %f] [G loss: %f]" % (epoch, discriminator_loss,
                                                    generator_loss))

            if epoch % img_frequency == 0:
                self._save_imgs(epoch)
                self._save_losses(losses)

    def _save_imgs(self, epoch):
        filenames = ['gan/%05d.png' % epoch, 'gan/last.png']
        generate_save_images(self._generator, 5, 5, self._latent_dim, filenames)

    @staticmethod
    def _save_losses(losses):
        filename = 'gan/losses.png'
        save_losses(losses, filename)