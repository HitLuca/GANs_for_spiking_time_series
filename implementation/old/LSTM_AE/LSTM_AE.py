import keras.backend as K
from keras import Input, Model, Sequential
from keras.layers import Lambda, LSTM, RepeatVector, Dense, TimeDistributed, Bidirectional
from keras.optimizers import Adam
import tensorflow as tf


class LSTM_AE:
    def __init__(self, timesteps, lstm_size, latent_dim):
        self._encoder = Sequential(name='encoder')
        self._encoder.add(Lambda(lambda x: K.expand_dims(x, -1), input_shape=(timesteps,)))
        self._encoder.add(Bidirectional(LSTM(lstm_size, return_sequences=False)))
        self._encoder.add(Dense(latent_dim, activation='tanh'))

        self._decoder = Sequential(name='decoder')
        self._decoder.add(RepeatVector(timesteps, input_shape=(latent_dim,)))
        self._decoder.add(Bidirectional(LSTM(lstm_size, return_sequences=True)))
        self._decoder.add(TimeDistributed(Dense(1, activation='tanh')))
        self._decoder.add(Lambda(lambda x: K.squeeze(x, -1)))

        self._autoencoder = Sequential([self._encoder, self._decoder], name='autoencoder')

    def get_model(self, lr=0.001):
        optimizer = Adam(lr=lr, epsilon=1e-08, amsgrad=True, clipnorm=1.0)
        self._autoencoder.compile(loss='mse', optimizer=optimizer)
        self._encoder.compile(loss='mse', optimizer=optimizer)
        self._decoder.compile(loss='mse', optimizer=optimizer)
        return self._autoencoder, self._encoder, self._decoder
