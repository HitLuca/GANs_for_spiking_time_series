from implementation.generative_models import utils
from keras import Model
from keras.layers import *
from keras.optimizers import *


class GAN:
    def __init__(self, timesteps, latent_dim, packing_degree, run_dir, img_dir, model_dir, generated_datesets_dir):
        self._timesteps = timesteps
        self._latent_dim = latent_dim
        self._packing_degree = packing_degree
        self._run_dir = run_dir
        self._img_dir = img_dir
        self._model_dir = model_dir
        self._generated_dataset_dir = generated_datesets_dir

        self._epoch = 0
        self._losses = [[], []]

    def build_model(self, generator_lr, discriminator_lr):
        self._generator = self._build_generator()
        self._discriminator = self._build_discriminator()

        z = Input(shape=(self._latent_dim,))
        input_samples = Input((self._timesteps,))
        expanded_input_samples = Lambda(lambda x: K.expand_dims(x, -1))(input_samples)

        supporting_inputs = Input((self._timesteps, self._packing_degree))

        utils.set_model_trainable(self._discriminator, False)

        generated_samples = self._generator(z)
        expanded_generatd_samples = Lambda(lambda x: K.expand_dims(x, -1))(generated_samples)
        merged_generated_samples = Concatenate(-1)([expanded_generatd_samples, supporting_inputs])
        
        generated_discriminated = self._discriminator(merged_generated_samples)
        
        self._generator_model = Model([z, supporting_inputs], generated_discriminated, 'generator_model')
        self._generator_model.compile(loss='binary_crossentropy', optimizer=RMSprop(generator_lr))
        
        utils.set_model_trainable(self._generator, False)
        utils.set_model_trainable(self._discriminator, True)
        
        merged_input_samples = Concatenate(-1)([expanded_input_samples, supporting_inputs])
        input_discriminated = self._discriminator(merged_input_samples)
        
        self._discriminator_model = Model([input_samples, supporting_inputs], input_discriminated)
        self._discriminator_model.compile(loss='binary_crossentropy', optimizer=RMSprop(discriminator_lr))

        return self._generator, self._discriminator, self._generator_model, self._discriminator_model

    def _build_generator(self):
        generator_inputs = Input((self._latent_dim,))
        generated = generator_inputs

        if self._latent_dim != 15:
            generated = Dense(15)(generated)
            generated = BatchNormalization()(generated)
            generated = LeakyReLU(0.2)(generated)

        generated = Lambda(lambda x: K.expand_dims(x))(generated)

        generated = Conv1D(64, 3, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        generated = UpSampling1D(2)(generated)

        generated = Conv1D(32, 3, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        generated = UpSampling1D(2)(generated)

        generated = Conv1D(16, 3, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        generated = UpSampling1D(2)(generated)

        generated = Conv1D(1, 3, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)

        generated = Lambda(lambda x: K.squeeze(x, -1))(generated)

        generated = Dense(self._timesteps, activation='tanh')(generated)

        generator = Model(generator_inputs, generated, 'generator')
        return generator

    def _build_discriminator(self):
        discriminator_inputs = Input((self._timesteps,))
        supporting_inputs = Input((self._timesteps, self._packing_degree))

        discriminated = discriminator_inputs

        discriminated = Lambda(lambda x: K.expand_dims(x))(discriminated)
        discriminated = Concatenate(-1)([discriminated, supporting_inputs])

        discriminated = Conv1D(16, 3, padding='same')(discriminated)
        discriminated = BatchNormalization()(discriminated)
        discriminated = LeakyReLU(0.2)(discriminated)
        discriminated = MaxPooling1D(2, padding='same')(discriminated)

        discriminated = Conv1D(32, 3, padding='same')(discriminated)
        discriminated = BatchNormalization()(discriminated)
        discriminated = LeakyReLU(0.2)(discriminated)
        discriminated = MaxPooling1D(2, padding='same')(discriminated)

        discriminated = Conv1D(64, 3, padding='same')(discriminated)
        discriminated = BatchNormalization()(discriminated)
        discriminated = LeakyReLU(0.2)(discriminated)
        discriminated = MaxPooling1D(2, padding='same')(discriminated)

        discriminated = Flatten()(discriminated)

        discriminated = Dense(15)(discriminated)
        discriminated = BatchNormalization()(discriminated)
        discriminated = LeakyReLU(0.2)(discriminated)

        discriminated = Dense(1, activation='sigmoid')(discriminated)

        discriminator = Model([discriminator_inputs, supporting_inputs], discriminated, 'discriminator')

        return discriminator

    def train(self, batch_size, epochs, n_generator, n_discriminator, dataset,
              img_frequency):
        half_batch = int(batch_size / 2)

        while self._epoch < epochs:
            self._epoch += 1
            discriminator_losses = []
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