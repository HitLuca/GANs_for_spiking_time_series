import keras
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop

from generative_models import utils


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


def build_critic(timesteps):
    kernel_initializer = keras.initializers.RandomNormal(0, 0.02)

    critic_inputs = Input((timesteps,))
    criticized = Lambda(lambda x: K.expand_dims(x, -1))(critic_inputs)

    criticized = Conv1D(32, 3, padding='same', kernel_initializer=kernel_initializer)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = MaxPooling1D(2, padding='same')(criticized)

    criticized = Conv1D(32, 3, padding='same', kernel_initializer=kernel_initializer)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = MaxPooling1D(2, padding='same')(criticized)

    criticized = Conv1D(32, 3, padding='same', kernel_initializer=kernel_initializer)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = MaxPooling1D(2, padding='same')(criticized)

    criticized = Conv1D(32, 3, padding='same', kernel_initializer=kernel_initializer)(criticized)
    criticized = LeakyReLU(0.2)(criticized)

    criticized = Flatten()(criticized)

    criticized = Dense(50, kernel_initializer=kernel_initializer)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = Dense(15, kernel_initializer=kernel_initializer)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = Dense(1, kernel_initializer=kernel_initializer)(criticized)

    critic = Model(critic_inputs, criticized, 'critic')

    return critic


def build_generator_model(generator, critic, generator_lr, latent_dim):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))

    generated_samples = generator(noise_samples)
    generated_criticized = critic(generated_samples)

    generator_model = Model(noise_samples, generated_criticized, 'generator_model')
    generator_model.compile(loss=utils.wasserstein_loss, optimizer=RMSprop(generator_lr))
    return generator_model


def build_critic_model(generator, critic, critic_lr, latent_dim, timesteps):
    utils.set_model_trainable(generator, False)
    utils.set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((timesteps,))

    generated_samples = generator(noise_samples)
    generated_criticized = critic(generated_samples)
    real_criticized = critic(real_samples)

    critic_model = Model([real_samples, noise_samples],
                         [real_criticized, generated_criticized], 'critic_model')
    critic_model.compile(loss=[utils.wasserstein_loss, utils.wasserstein_loss], optimizer=RMSprop(critic_lr),
                         loss_weights=[1 / 2, 1 / 2])
    return critic_model


def clip_weights(model, clip_value):
    for l in model.layers:
        weights = [np.clip(w, -clip_value, clip_value) for w in l.get_weights()]
        l.set_weights(weights)
