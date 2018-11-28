from functools import partial

from keras import Model
from keras.layers import *
from keras.layers.merge import _Merge
from keras.optimizers import Adam

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


def build_critic(timesteps, use_mbd, use_packing, packing_degree):
    if use_packing:
        critic_inputs = Input((timesteps, packing_degree + 1))
        criticized = critic_inputs
    else:
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
    if use_mbd:
        criticized = utils.MinibatchDiscrimination(15, 3)(criticized)

    criticized = Dense(50)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = Dense(15)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = Dense(1)(criticized)

    critic = Model(critic_inputs, criticized, 'critic')

    return critic


def build_generator_model(generator, critic, latent_dim, timesteps, use_packing, packing_degree, batch_size,
                          generator_lr):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))
    generated_samples = generator(noise_samples)

    if use_packing:
        generated_samples = Lambda(lambda x: K.reshape(x, (batch_size, timesteps, 1)))(generated_samples)
        supporting_noise_samples = Input((latent_dim, packing_degree))

        reshaped_supporting_noise_samples = Lambda(
            lambda x: K.reshape(x, (batch_size * packing_degree, latent_dim)))(
            supporting_noise_samples)

        supporting_generated_samples = generator(reshaped_supporting_noise_samples)
        supporting_generated_samples = Lambda(
            lambda x: K.reshape(x, (batch_size, timesteps, packing_degree)))(
            supporting_generated_samples)
        merged_generated_samples = Concatenate(-1)([generated_samples, supporting_generated_samples])

        generated_criticized = critic(merged_generated_samples)

        generator_model = Model([noise_samples, supporting_noise_samples], generated_criticized, 'generator_model')
        generator_model.compile(optimizer=Adam(generator_lr, beta_1=0, beta_2=0.9), loss=utils.wasserstein_loss)
    else:
        generated_criticized = critic(generated_samples)

        generator_model = Model([noise_samples], generated_criticized, 'generator_model')
        generator_model.compile(optimizer=Adam(generator_lr, beta_1=0, beta_2=0.9), loss=utils.wasserstein_loss)
    return generator_model


def build_critic_model(generator, critic, latent_dim, timesteps, use_packing, packing_degree, batch_size, critic_lr,
                       gradient_penality_weight):
    utils.set_model_trainable(generator, False)
    utils.set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((timesteps,))

    if use_packing:
        supporting_noise_samples = Input((latent_dim, packing_degree))
        supporting_real_samples = Input((timesteps, packing_degree))

        reshaped_supporting_noise_samples = Lambda(
            lambda x: K.reshape(x, (batch_size * packing_degree, latent_dim)))(
            supporting_noise_samples)
        generated_samples = generator(noise_samples)
        supporting_generated_samples = generator(reshaped_supporting_noise_samples)

        expanded_generated_samples = Lambda(lambda x: K.reshape(x, (batch_size, timesteps, 1)))(
            generated_samples)
        expanded_generated_supporting_samples = Lambda(
            lambda x: K.reshape(x, (batch_size, timesteps, packing_degree)))(
            supporting_generated_samples)

        merged_generated_samples = Concatenate(-1)([expanded_generated_samples, expanded_generated_supporting_samples])

        generated_criticized = critic(merged_generated_samples)

        expanded_real_samples = Lambda(lambda x: K.reshape(x, (batch_size, timesteps, 1)))(real_samples)
        merged_real_samples = Lambda(lambda x: K.concatenate(x, -1))([expanded_real_samples, supporting_real_samples])

        real_criticized = critic(merged_real_samples)

        averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples])

        expanded_averaged_samples = Lambda(lambda x: K.reshape(x, (batch_size, timesteps, 1)))(
            averaged_samples)

        expanded_supporting_real_samples = Lambda(
            lambda x: K.reshape(x, ((batch_size * packing_degree), timesteps)))(
            supporting_real_samples)
        averaged_support_samples = RandomWeightedAverage((batch_size * packing_degree))(
            [expanded_supporting_real_samples, supporting_generated_samples])

        averaged_support_samples = Lambda(
            lambda x: K.reshape(x, (batch_size, timesteps, packing_degree)))(
            averaged_support_samples)

        merged_averaged_samples = Concatenate(-1)([expanded_averaged_samples, averaged_support_samples])

        averaged_criticized = critic(merged_averaged_samples)

        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=merged_averaged_samples,
                                  gradient_penalty_weight=gradient_penality_weight)
        partial_gp_loss.__name__ = 'gradient_penalty'

        critic_model = Model([real_samples, noise_samples, supporting_real_samples, supporting_noise_samples],
                             [real_criticized, generated_criticized, averaged_criticized], 'critic_model')

        critic_model.compile(optimizer=Adam(critic_lr, beta_1=0, beta_2=0.9),
                             loss=[utils.wasserstein_loss, utils.wasserstein_loss, partial_gp_loss],
                             loss_weights=[1 / 3, 1 / 3, 1 / 3])
    else:
        generated_samples = generator(noise_samples)
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
                             loss=[utils.wasserstein_loss, utils.wasserstein_loss, partial_gp_loss],
                             loss_weights=[1 / 3, 1 / 3, 1 / 3])
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
