import sys
from functools import partial

from keras import Model
from keras.layers import *
from keras.layers.merge import _Merge
from keras.losses import mean_squared_error
from keras.optimizers import Adam

sys.path.append("..")
import utils


def build_encoder(latent_dim, timesteps):
    encoder_inputs = Input((timesteps,))
    encoded = Lambda(lambda x: K.expand_dims(x, -1))(encoder_inputs)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    # encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    # encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    # encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    # encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)

    encoded = Flatten()(encoded)

    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)

    encoder = Model(encoder_inputs, [z_mean, z_log_var])
    return encoder


def build_decoder(latent_dim, timesteps):
    decoder_inputs = Input((latent_dim,))
    decoded = decoder_inputs

    decoded = Dense(15)(decoded)
    # decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)

    decoded = Lambda(lambda x: K.expand_dims(x))(decoded)

    decoded = Conv1D(32, 3, padding='same')(decoded)
    # decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)
    decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(32, 3, padding='same')(decoded)
    # decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)
    decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(32, 3, padding='same')(decoded)
    # decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)
    decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(1, 3, padding='same')(decoded)
    # decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)

    decoded = Lambda(lambda x: K.squeeze(x, -1))(decoded)

    decoded = Dense(timesteps, activation='tanh')(decoded)

    decoder = Model(decoder_inputs, decoded, 'decoder')
    return decoder


def build_critic(timesteps):
    critic_inputs = Input((timesteps,))
    criticized = Lambda(lambda x: K.expand_dims(x, -1))(critic_inputs)

    criticized = Conv1D(32, 3, padding='same')(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = MaxPooling1D(2, padding='same')(criticized)

    criticized = Conv1D(32, 3, padding='same')(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = MaxPooling1D(2, padding='same')(criticized)

    criticized = Conv1D(32, 3, padding='same')(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = MaxPooling1D(2, padding='same')(criticized)

    criticized = Conv1D(32, 3, padding='same')(criticized)
    criticized = LeakyReLU(0.2)(criticized)

    criticized = Flatten()(criticized)

    critic_hidden = Model(critic_inputs, criticized, 'critic_hidden')

    criticized = Dense(50)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = Dense(15)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = Dense(1)(criticized)

    critic = Model(critic_inputs, criticized, 'critic')

    return critic, critic_hidden


def build_vae_model(encoder, decoder_generator, critic_hidden, critic, latent_dim, timesteps, vae_lr):
    utils.set_model_trainable(encoder, True)
    utils.set_model_trainable(decoder_generator, True)
    utils.set_model_trainable(critic_hidden, False)

    real_samples = Input((timesteps,))
    noise_samples = Input((latent_dim,))

    generated_samples = decoder_generator(noise_samples)
    generated_criticized = critic(generated_samples)

    z_mean, z_log_var = encoder(real_samples)

    sampled_z = Lambda(sampling)([z_mean, z_log_var])
    decoded_inputs = decoder_generator(sampled_z)

    real_criticized = critic_hidden(real_samples)
    decoded_criticized = critic_hidden(decoded_inputs)

    generator_vae_model = Model([real_samples, noise_samples], [generated_criticized])
    generator_vae_model.compile(optimizer=Adam(lr=vae_lr, beta_1=0, beta_2=0.9), loss=custom_loss(z_mean, z_log_var, real_criticized, decoded_criticized))

    generator_model = Model(noise_samples, generated_samples)
    return generator_vae_model, generator_model


def custom_loss(z_mean, z_log_var, real_criticized, decoded_criticized):
    def loss(y_true, y_pred):
        gan_loss = utils.wasserstein_loss(y_true, y_pred)
        xent_loss = mean_squared_error(real_criticized, decoded_criticized)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        vae_loss = xent_loss + kl_loss
        return 0.4 * gan_loss + (1-0.4) * vae_loss
    return loss


# def build_generator_model(encoder, decoder_generator, critic, latent_dim, generator_lr):
#     utils.set_model_trainable(encoder, False)
#     utils.set_model_trainable(decoder_generator, True)
#     utils.set_model_trainable(critic, False)
#
#     noise_samples = Input((latent_dim,))
#
#     generated_samples = decoder_generator(noise_samples)
#     generated_criticized = critic(generated_samples)
#
#     generator_model = Model(noise_samples, generated_criticized)
#     generator_model.compile(optimizer=Adam(lr=generator_lr), loss=utils.wasserstein_loss)
#
#     generator = Model(noise_samples, generated_samples)
#
#     return generator_model, generator


def build_critic_model(encoder, decoder_generator, critic, latent_dim, timesteps, batch_size, critic_lr, gradient_penality_weight):
    utils.set_model_trainable(encoder, False)
    utils.set_model_trainable(decoder_generator, False)
    utils.set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((timesteps,))

    generated_samples = decoder_generator(noise_samples)

    generated_criticized = critic(generated_samples)
    real_criticized = critic(real_samples)

    averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples])
    averaged_criticized = critic(averaged_samples)

    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=gradient_penality_weight)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model([real_samples, noise_samples],
                         [real_criticized, generated_criticized, averaged_criticized], 'critic_model')

    critic_model.compile(optimizer=Adam(critic_lr, beta_1=0, beta_2=0.9),
                         loss=[utils.wasserstein_loss, utils.wasserstein_loss, partial_gp_loss])
    return critic_model


def gradient_penalty_loss(_, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = batch_size

    def _merge_function(self, inputs):
        weights = K.random_uniform((self._batch_size, 1))
        averaged_inputs = (weights * inputs[0]) + ((1 - weights) * inputs[1])
        return averaged_inputs


def sampling(args):
    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(z_log_var) * epsilon
