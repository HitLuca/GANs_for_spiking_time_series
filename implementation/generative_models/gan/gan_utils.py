from keras import Model
from keras.layers import *
from keras.optimizers import Adam
import sys
sys.path.append("..")
import utils


def build_generator(latent_dim, timesteps):
    generator_inputs = Input((latent_dim,))

    generated = generator_inputs

    generated = Dense(50, activation='relu')(generated)
    generated = Dense(timesteps, activation='tanh')(generated)
    # generated = Lambda(lambda x: K.expand_dims(x))(generator_inputs)

    # generated = Conv1D(32, 3, padding='same')(generated)
    # # generated = utils.BatchNormalizationGAN()(generated)
    # generated = LeakyReLU(0.2)(generated)
    # generated = UpSampling1D(2)(generated)
    #
    # generated = Conv1D(32, 3, padding='same')(generated)
    # # generated = utils.BatchNormalizationGAN()(generated)
    # generated = LeakyReLU(0.2)(generated)
    # generated = UpSampling1D(2)(generated)
    #
    # generated = Conv1D(32, 3, padding='same')(generated)
    # # generated = utils.BatchNormalizationGAN()(generated)
    # generated = LeakyReLU(0.2)(generated)
    # generated = UpSampling1D(2)(generated)
    #
    # generated = Conv1D(1, 3, padding='same')(generated)
    # # generated = utils.BatchNormalizationGAN()(generated)
    # generated = LeakyReLU(0.2)(generated)
    #
    # generated = Lambda(lambda x: K.squeeze(x, -1))(generated)
    #
    # generated = Dense(timesteps, activation='tanh')(generated)

    generator = Model(generator_inputs, generated, 'generator')
    return generator


def build_discriminator(timesteps, use_mbd, use_packing, packing_degree):
    if use_packing:
        discriminator_inputs = Input((timesteps, packing_degree + 1))
        discriminated = discriminator_inputs
    else:
        discriminator_inputs = Input((timesteps,))

    discriminated = discriminator_inputs
    discriminated = Dense(50, activation='relu')(discriminated)
    discriminated = Dense(1, activation='sigmoid')(discriminated)
    #     discriminated = Lambda(lambda x: K.expand_dims(x, -1))(discriminator_inputs)
    #
    # discriminated = Conv1D(32, 3, padding='same')(discriminated)
    # discriminated = LeakyReLU(0.2)(discriminated)
    # discriminated = MaxPooling1D(2, padding='same')(discriminated)
    #
    # discriminated = Conv1D(32, 3, padding='same')(discriminated)
    # discriminated = LeakyReLU(0.2)(discriminated)
    # discriminated = MaxPooling1D(2, padding='same')(discriminated)
    #
    # discriminated = Conv1D(32, 3, padding='same')(discriminated)
    # discriminated = LeakyReLU(0.2)(discriminated)
    # discriminated = MaxPooling1D(2, padding='same')(discriminated)
    #
    # discriminated = Conv1D(32, 3, padding='same')(discriminated)
    # discriminated = LeakyReLU(0.2)(discriminated)
    #
    # discriminated = Flatten()(discriminated)
    # if use_mbd:
    #     discriminated = utils.MinibatchDiscrimination(15, 3)(discriminated)
    #
    # discriminated = Dense(50)(discriminated)
    # discriminated = LeakyReLU(0.2)(discriminated)
    # discriminated = Dense(15)(discriminated)
    # discriminated = LeakyReLU(0.2)(discriminated)
    # discriminated = Dense(1, activation='sigmoid')(discriminated)

    discriminator = Model(discriminator_inputs, discriminated, 'discriminator')

    return discriminator


def build_generator_model(generator, discriminator, latent_dim, timesteps, use_packing, packing_degree, batch_size, generator_lr):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(discriminator, False)

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

        generated_discriminated = discriminator(merged_generated_samples)

        generator_model = Model([noise_samples, supporting_noise_samples], generated_discriminated, 'generator_model')
        generator_model.compile(optimizer=Adam(generator_lr), loss='binary_crossentropy')
    else:
        generated_discriminated = discriminator(generated_samples)

        generator_model = Model([noise_samples], generated_discriminated, 'generator_model')
        generator_model.compile(optimizer=Adam(generator_lr), loss='binary_crossentropy')
    return generator_model


def build_discriminator_model(generator, discriminator, latent_dim, timesteps, use_packing, packing_degree, batch_size, discriminator_lr):
    utils.set_model_trainable(generator, False)
    utils.set_model_trainable(discriminator, True)

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

        generated_discriminated = discriminator(merged_generated_samples)

        expanded_real_samples = Lambda(lambda x: K.reshape(x, (batch_size, timesteps, 1)))(real_samples)
        merged_real_samples = Lambda(lambda x: K.concatenate(x, -1))([expanded_real_samples, supporting_real_samples])

        real_discriminated = discriminator(merged_real_samples)

        discriminator_model = Model([real_samples, noise_samples, supporting_real_samples, supporting_noise_samples],
                             [real_discriminated, generated_discriminated], 'discriminator_model')
        discriminator_model.compile(optimizer=Adam(discriminator_lr), loss=['binary_crossentropy', 'binary_crossentropy'])
    else:
        generated_samples = generator(noise_samples)
        generated_discriminated = discriminator(generated_samples)
        real_discriminated = discriminator(real_samples)

        discriminator_model = Model([real_samples, noise_samples],
                             [real_discriminated, generated_discriminated], 'discriminator_model')
        discriminator_model.compile(optimizer=Adam(discriminator_lr), loss=['binary_crossentropy', 'binary_crossentropy'])
    return discriminator_model
