from keras import Model
from keras.layers import *
from keras.losses import mean_squared_error
from keras.optimizers import Adam


def build_encoder(latent_dim, timesteps):
    encoder_inputs = Input((timesteps,))
    encoded = Lambda(lambda x: K.expand_dims(x, -1))(encoder_inputs)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)
    encoded = MaxPooling1D(2, padding='same')(encoded)

    encoded = Conv1D(32, 3, padding='same')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)

    encoded = Flatten()(encoded)
    
    encoded = Dense(50)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)

    encoded = Dense(latent_dim)(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = LeakyReLU(0.2)(encoded)

    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)

    encoder = Model(encoder_inputs, [z_mean, z_log_var])
    return encoder
    

def build_decoder(latent_dim, timesteps):
    decoder_inputs = Input((latent_dim,))
    decoded = decoder_inputs

    decoded = Dense(15)(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)

    decoded = Lambda(lambda x: K.expand_dims(x))(decoded)

    decoded = Conv1D(32, 3, padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)
    decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(32, 3, padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)
    decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(32, 3, padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)
    decoded = UpSampling1D(2)(decoded)

    decoded = Conv1D(1, 3, padding='same')(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)

    decoded = Lambda(lambda x: K.squeeze(x, -1))(decoded)

    decoded = Dense(timesteps, activation='tanh')(decoded)

    decoder = Model(decoder_inputs, decoded, 'decoder')
    return decoder


def build_vae_model(encoder, decoder, latent_dim, timesteps, batch_size, lr):
    inputs = Input((timesteps,))
    z = Input((latent_dim,))

    z_mean, z_log_var = encoder(inputs)

    sampled_z = Lambda(sampling, arguments={'batch_size':batch_size, 'latent_dim':latent_dim})([z_mean, z_log_var])
    decoded_inputs = decoder(sampled_z)

    vae_model = Model(inputs, decoded_inputs)
    vae_model.compile(optimizer=Adam(lr=lr), loss=vae_loss(z_mean, z_log_var))

    generator = Model(z, decoder(z))
    return vae_model, generator


def sampling(args, batch_size, latent_dim):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_var / 2.0) * epsilon


def vae_loss(z_mean, z_log_var):
    def loss(y_true, y_pred):
        xent_loss = mean_squared_error(y_true, y_pred)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
        return xent_loss + kl_loss
    return loss