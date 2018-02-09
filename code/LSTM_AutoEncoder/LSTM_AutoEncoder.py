from datetime import datetime
import os
import keras
import keras.backend as K
import tensorflow as tf
from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Lambda, LSTM, RepeatVector, Dense, TimeDistributed, Bidirectional, Concatenate
from matplotlib import pyplot as plt
from scipy import io
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class LSTM_AutoEncoder:
    def __init__(self, timesteps, latent_dim, reverse):
        model_inputs = Input(shape=(timesteps,))
        expanded_inputs = Lambda(lambda x: K.expand_dims(x, -1))(model_inputs)
        encoder_outputs, encoder_h, encoder_c = LSTM(latent_dim, return_sequences=False, return_state=True)(
            expanded_inputs)
        encoder_state = [encoder_h, encoder_c]

        decoder_inputs = Input(shape=(1,))
        repeated_inputs = RepeatVector(timesteps)(decoder_inputs)
        decoder_outputs = LSTM(latent_dim, return_sequences=True, go_backwards=reverse)(repeated_inputs, initial_state=encoder_state)
        decoder_outputs = TimeDistributed(Dense(1, activation='sigmoid'))(decoder_outputs)

        model_outputs = Lambda(lambda x: K.squeeze(x, -1))(decoder_outputs)

        self._sequence_autoencoder = Model([model_inputs, decoder_inputs], model_outputs)

    def get_model(self, lr):
        optimizer = keras.optimizers.Adam(lr=lr, epsilon=1e-08, amsgrad=True)
        self._sequence_autoencoder.compile(loss='mse', optimizer=optimizer)
        return self._sequence_autoencoder
