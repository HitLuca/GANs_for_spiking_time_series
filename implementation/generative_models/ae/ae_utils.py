import sys
from enum import Enum

from keras import Model
from keras.layers import *
from keras.optimizers import Adam

sys.path.append("..")
import utils


def build_encoder(latent_dim, timesteps, encoder_type):
    encoder_inputs = Input((timesteps,))
    encoded = encoder_inputs

    if encoder_type == AE_type.dense.name:
        encoded = Dense(50)(encoded)
        encoded = utils.BatchNormalization()(encoded)
        encoded = LeakyReLU(0.2)(encoded)
        encoded = Dense(50)(encoded)
        encoded = utils.BatchNormalization()(encoded)
        encoded = LeakyReLU(0.2)(encoded)
        encoded = Dense(latent_dim)(encoded)
        encoded = utils.BatchNormalization()(encoded)
        encoded = LeakyReLU(0.2)(encoded)

    elif encoder_type == AE_type.lstm.name:
        encoded = Lambda(lambda x: K.expand_dims(x, -1))(encoded)

        encoded = LSTM(32, return_sequences=False)(encoded)
        encoded = Dense(latent_dim)(encoded)
        encoded = utils.BatchNormalization()(encoded)
        encoded = LeakyReLU(0.2)(encoded)

    elif encoder_type == AE_type.blstm.name:
        encoded = Lambda(lambda x: K.expand_dims(x, -1))(encoded)

        encoded = Bidirectional(LSTM(32, return_sequences=False))(encoded)
        encoded = Dense(latent_dim)(encoded)
        encoded = utils.BatchNormalization()(encoded)
        encoded = LeakyReLU(0.2)(encoded)

    elif encoder_type == AE_type.cnn.name:
        encoded = Lambda(lambda x: K.expand_dims(x))(encoded)

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
        encoded = Dense(latent_dim)(encoded)
        encoded = utils.BatchNormalization()(encoded)
        encoded = LeakyReLU(0.2)(encoded)

    encoder = Model(encoder_inputs, encoded, 'encoder')
    return encoder


def build_decoder(latent_dim, timesteps, decoder_type):
    decoder_inputs = Input((latent_dim,))
    decoded = decoder_inputs

    decoded = Dense(15)(decoded)
    decoded = utils.BatchNormalization()(decoded)
    decoded = LeakyReLU(0.2)(decoded)

    if decoder_type == AE_type.dense.name:
        decoded = Dense(50)(decoded)
        decoded = utils.BatchNormalization()(decoded)
        decoded = LeakyReLU(0.2)(decoded)
        decoded = Dense(50)(decoded)
        decoded = utils.BatchNormalization()(decoded)
        decoded = LeakyReLU(0.2)(decoded)
        decoded = Dense(timesteps, activation='tanh')(decoded)

    elif decoder_type == AE_type.lstm.name:
        decoded = RepeatVector(timesteps)(decoded)
        decoded = LSTM(32, return_sequences=True)(decoded)
        decoded = TimeDistributed(Dense(1, activation='tanh'))(decoded)
        decoded = Lambda(lambda x: K.squeeze(x, -1))(decoded)

    elif decoder_type == AE_type.blstm.name:
        decoded = RepeatVector(timesteps)(decoded)
        decoded = Bidirectional(LSTM(32, return_sequences=True))(decoded)
        decoded = TimeDistributed(Dense(1, activation='tanh'))(decoded)
        decoded = Lambda(lambda x: K.squeeze(x, -1))(decoded)

    elif decoder_type == AE_type.cnn.name:
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


def build_ae_model(encoder, decoder, latent_dim, timesteps, lr):
    inputs = Input((timesteps,))
    z = Input((latent_dim,))

    encoded_inputs = encoder(inputs)
    decoded_inputs = decoder(encoded_inputs)

    ae_model = Model(inputs, decoded_inputs)
    ae_model.compile(loss='mse', optimizer=Adam(lr=lr))

    generator = Model(z, decoder(z))

    return ae_model, generator


class AE_type(Enum):
    dense = 1
    lstm = 2
    blstm = 3
    cnn = 4
