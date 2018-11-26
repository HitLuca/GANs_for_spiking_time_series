from keras import Model
from keras.layers import *
from keras.losses import mean_squared_error
from keras.optimizers import Adam
from generative_models import utils


def build_encoder(latent_dim, timesteps):
    encoder_inputs = Input((timesteps,))
    encoded = Lambda(lambda x: K.expand_dims(x, -1))(encoder_inputs)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    encoded = utils.BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    encoded = utils.BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    encoded = utils.BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    encoded = utils.BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)

    encoded = Flatten()(encoded)

    encoded = Dense(128)(encoded)
    encoded = utils.BatchNormalization()(encoded)
    encoded = Activation('tanh')(encoded)

    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)

    encoder = Model(encoder_inputs, [z_mean, z_log_var])
    return encoder


def build_decoder(latent_dim, timesteps):
    decoder_inputs = Input((latent_dim,))
    decoded = decoder_inputs

    decoded = Dense(15)(decoded)
    decoded = utils.BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)

    decoded = Lambda(lambda x: K.expand_dims(x))(decoded)

    decoded = Conv1D(32, 3, padding='same')(decoded)
    decoded = utils.BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)
    decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(32, 3, padding='same')(decoded)
    decoded = utils.BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)
    decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(32, 3, padding='same')(decoded)
    decoded = utils.BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)
    decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(1, 3, padding='same')(decoded)
    decoded = utils.BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)

    decoded = Lambda(lambda x: K.squeeze(x, -1))(decoded)

    decoded = Dense(timesteps, activation='tanh')(decoded)

    decoder = Model(decoder_inputs, decoded, 'decoder')
    return decoder


def build_vae_model(encoder, decoder, latent_dim, timesteps, lr):
    inputs = Input((timesteps,))
    z = Input((latent_dim,))

    z_mean, z_log_var = encoder(inputs)

    sampled_z = Lambda(sampling)([z_mean, z_log_var])
    decoded_inputs = decoder(sampled_z)

    vae_model = Model(inputs, decoded_inputs)
    vae_model.compile(optimizer=Adam(lr=lr), loss=vae_loss(z_mean, z_log_var, timesteps))

    generator = Model(z, decoder(z))
    return vae_model, generator


def sampling(args):
    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def vae_loss(z_mean, z_log_var, timesteps):
    def loss(y_true, y_pred):
        mse_loss = mean_squared_error(y_true, y_pred)
        mse_loss *= timesteps
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(mse_loss + kl_loss)

    return loss
