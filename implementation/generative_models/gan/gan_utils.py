import sys

from keras import Model
from keras.layers import *
from keras.optimizers import Adam, RMSprop

sys.path.append("..")
import utils


def build_generator(latent_dim, timesteps):
    generator_inputs = Input((latent_dim,))
    generated = generator_inputs

    generated = Dense(15)(generated)
    generated = utils.BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)

    generated = Lambda(lambda x: K.expand_dims(x))(generated)

    generated = Conv1D(32, 3, padding='same')(generated)
    generated = utils.BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)
    generated = UpSampling1D(2)(generated)

    generated = Conv1D(32, 3, padding='same')(generated)
    generated = utils.BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)
    generated = UpSampling1D(2)(generated)

    generated = Conv1D(32, 3, padding='same')(generated)
    generated = utils.BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)
    generated = UpSampling1D(2)(generated)

    generated = Conv1D(1, 3, padding='same')(generated)
    generated = utils.BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)

    generated = Lambda(lambda x: K.squeeze(x, -1))(generated)

    generated = Dense(timesteps, activation='tanh')(generated)

    generator = Model(generator_inputs, generated, 'generator')
    return generator


def build_discriminator(timesteps):
    discriminator_inputs = Input((timesteps,))
    discriminated = Lambda(lambda x: K.expand_dims(x, -1))(discriminator_inputs)

    discriminated = Conv1D(32, 3, padding='same')(discriminated)
    discriminated = LeakyReLU(0.2)(discriminated)
    discriminated = MaxPooling1D(2, padding='same')(discriminated)

    discriminated = Conv1D(32, 3, padding='same')(discriminated)
    discriminated = LeakyReLU(0.2)(discriminated)
    discriminated = MaxPooling1D(2, padding='same')(discriminated)

    discriminated = Conv1D(32, 3, padding='same')(discriminated)
    discriminated = LeakyReLU(0.2)(discriminated)
    discriminated = MaxPooling1D(2, padding='same')(discriminated)

    discriminated = Conv1D(32, 3, padding='same')(discriminated)
    discriminated = LeakyReLU(0.2)(discriminated)

    discriminated = Flatten()(discriminated)

    discriminated = Dense(50)(discriminated)
    discriminated = LeakyReLU(0.2)(discriminated)
    discriminated = Dense(15)(discriminated)
    discriminated = LeakyReLU(0.2)(discriminated)
    discriminated = Dense(1, activation='sigmoid')(discriminated)

    discriminator = Model(discriminator_inputs, discriminated, 'discriminator')

    return discriminator


def build_generator_model(generator, discriminator, latent_dim, generator_lr):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(discriminator, False)

    noise_samples = Input((latent_dim,))
    generated_samples = generator(noise_samples)

    generated_discriminated = discriminator(generated_samples)

    generator_model = Model([noise_samples], generated_discriminated, 'generator_model')
    generator_model.compile(optimizer=Adam(generator_lr), loss='binary_crossentropy')
    return generator_model


def build_discriminator_model(generator, discriminator, latent_dim, timesteps, discriminator_lr):
    utils.set_model_trainable(generator, False)
    utils.set_model_trainable(discriminator, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((timesteps,))

    generated_samples = generator(noise_samples)
    generated_discriminated = discriminator(generated_samples)
    real_discriminated = discriminator(real_samples)

    discriminator_model = Model([real_samples, noise_samples],
                                [real_discriminated, generated_discriminated], 'discriminator_model')
    discriminator_model.compile(optimizer=RMSprop(discriminator_lr),
                                loss=['binary_crossentropy', 'binary_crossentropy'],
                                loss_weights=[1 / 2, 1 / 2])
    return discriminator_model
