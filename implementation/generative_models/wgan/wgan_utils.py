from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop
import sys
sys.path.append("..")
import utils


def build_generator(latent_dim, timesteps, kernel_initializer):
    generator_inputs = Input((latent_dim,))
    generated = generator_inputs

    generated = Dense(15, kernel_initializer=kernel_initializer)(generated)
    generated = BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)

    generated = Lambda(lambda x: K.expand_dims(x))(generated)

    generated = Conv1D(32, 3, padding='same', kernel_initializer=kernel_initializer)(generated)
    generated = BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)
    generated = UpSampling1D(2)(generated)

    generated = Conv1D(32, 3, padding='same', kernel_initializer=kernel_initializer)(generated)
    generated = BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)
    generated = UpSampling1D(2)(generated)

    generated = Conv1D(32, 3, padding='same', kernel_initializer=kernel_initializer)(generated)
    generated = BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)
    generated = UpSampling1D(2)(generated)

    generated = Conv1D(1, 3, padding='same', kernel_initializer=kernel_initializer)(generated)
    generated = BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)

    generated = Lambda(lambda x: K.squeeze(x, -1))(generated)

    generated = Dense(timesteps, activation='tanh', kernel_initializer=kernel_initializer)(generated)

    generator = Model(generator_inputs, generated, 'generator')
    return generator


def build_critic(timesteps, weights_initializer, use_mbd, use_packing, packing_degree):
    if use_packing:
        critic_inputs = Input((timesteps, packing_degree + 1))
        criticized = critic_inputs
    else:
        critic_inputs = Input((timesteps,))
        criticized = Lambda(lambda x: K.expand_dims(x, -1))(critic_inputs)

    criticized = Conv1D(32, 3, padding='same', kernel_initializer=weights_initializer)(criticized)
    criticized = BatchNormalization()(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = MaxPooling1D(2, padding='same')(criticized)

    criticized = Conv1D(32, 3, padding='same', kernel_initializer=weights_initializer)(criticized)
    criticized = BatchNormalization()(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = MaxPooling1D(2, padding='same')(criticized)

    criticized = Conv1D(32, 3, padding='same', kernel_initializer=weights_initializer)(criticized)
    criticized = BatchNormalization()(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = MaxPooling1D(2, padding='same')(criticized)

    criticized = Conv1D(32, 3, padding='same', kernel_initializer=weights_initializer)(criticized)
    criticized = BatchNormalization()(criticized)
    criticized = LeakyReLU(0.2)(criticized)

    criticized = Flatten()(criticized)
    if use_mbd:
        criticized = utils.MinibatchDiscrimination(15, 3)(criticized)

    criticized = Dense(50, kernel_initializer=weights_initializer)(criticized)
    criticized = BatchNormalization()(criticized)
    criticized = LeakyReLU(0.2)(criticized)

    criticized = Dense(15, kernel_initializer=weights_initializer)(criticized)
    criticized = BatchNormalization()(criticized)
    criticized = LeakyReLU(0.2)(criticized)

    criticized = Dense(1, kernel_initializer=weights_initializer)(criticized)

    critic = Model(critic_inputs, criticized, 'critic')

    return critic


def build_generator_model(generator, critic, generator_lr, latent_dim, batch_size, timesteps, use_packing, packing_degree):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))

    if use_packing:
        supporting_noise_samples = Input((latent_dim, packing_degree))

        reshaped_supporting_noise_samples = Lambda(lambda x: K.reshape(x, (batch_size * packing_degree, latent_dim)))(supporting_noise_samples)
        generated_samples = generator(noise_samples)
        generated_samples = Lambda(lambda x: K.reshape(x, (batch_size, timesteps, 1)))(
            generated_samples)
        supporting_generated_samples = generator(reshaped_supporting_noise_samples)
        supporting_generated_samples = Lambda(lambda x: K.reshape(x, (batch_size, timesteps, packing_degree)))(
            supporting_generated_samples)

        merged_generated_samples = Lambda(lambda x: K.concatenate(x, -1))([generated_samples, supporting_generated_samples])

        generated_criticized = critic(merged_generated_samples)

        generator_model = Model([noise_samples, supporting_noise_samples], generated_criticized, 'generator_model')
        generator_model.compile(loss=utils.wasserstein_loss, optimizer=RMSprop(generator_lr))
    else:
        generated_samples = generator(noise_samples)
        generated_criticized = critic(generated_samples)

        generator_model = Model([noise_samples], generated_criticized, 'generator_model')
        generator_model.compile(loss=utils.wasserstein_loss, optimizer=RMSprop(generator_lr))
    return generator_model


def build_critic_model(generator, critic, critic_lr, latent_dim, batch_size, timesteps, use_packing, packing_degree):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))
    real_samples = Input((timesteps,))

    if use_packing:
        supporting_noise_samples = Input((latent_dim, packing_degree))
        supporting_real_samples = Input((timesteps, packing_degree))

        reshaped_supporting_noise_samples = Lambda(lambda x: K.reshape(x, (batch_size * packing_degree, latent_dim)))(supporting_noise_samples)
        generated_samples = generator(noise_samples)
        generated_supporting_samples = generator(reshaped_supporting_noise_samples)

        expanded_generated_samples = Lambda(lambda x: K.reshape(x, (batch_size, timesteps, 1)))(generated_samples)
        expanded_generated_supporting_samples = Lambda(
            lambda x: K.reshape(x, (batch_size, timesteps, packing_degree)))(generated_supporting_samples)
        merged_generated_samples = Lambda(lambda x: K.concatenate(x, -1))(
            [expanded_generated_samples, expanded_generated_supporting_samples])

        generated_criticized = critic(merged_generated_samples)

        expanded_real_samples = Lambda(lambda x: K.reshape(x, (batch_size, timesteps, 1)))(real_samples)
        merged_real_samples = Lambda(lambda x: K.concatenate(x, -1))([expanded_real_samples, supporting_real_samples])

        real_criticized = critic(merged_real_samples)

        critic_model = Model([real_samples, noise_samples, supporting_real_samples, supporting_noise_samples],
                             [real_criticized, generated_criticized], 'critic_model')
        critic_model.compile(loss=[utils.wasserstein_loss, utils.wasserstein_loss], optimizer=RMSprop(critic_lr))
    else:
        generated_samples = generator(noise_samples)
        generated_criticized = critic(generated_samples)
        real_criticized = critic(real_samples)

        critic_model = Model([real_samples, noise_samples],
                             [real_criticized, generated_criticized], 'critic_model')
        critic_model.compile(loss=[utils.wasserstein_loss, utils.wasserstein_loss], optimizer=RMSprop(critic_lr))
    return critic_model


def clip_weights(model, clip_value):
    for l in model.layers:
        weights = [np.clip(w, -clip_value, clip_value) for w in l.get_weights()]
        l.set_weights(weights)