import keras.backend as K
from keras import Input, Model
from keras.layers import Lambda, LSTM, RepeatVector, Dense, TimeDistributed, Bidirectional
from keras.optimizers import Adam


class LSTM_AE:
    def __init__(self, timesteps, lstm_size, latent_dim):
        model_inputs = Input(shape=(timesteps,))

        encoder = Lambda(lambda x: K.expand_dims(x, -1))(model_inputs)
        encoder = Bidirectional(LSTM(lstm_size, return_sequences=False))(encoder)
        encoder = Dense(latent_dim, activation='tanh', name='embedding')(encoder)

        decoder = RepeatVector(timesteps)(encoder)
        decoder = Bidirectional(LSTM(lstm_size, return_sequences=True))(decoder)
        decoder = TimeDistributed(Dense(1, activation='tanh'))(decoder)

        model_outputs = Lambda(lambda x: K.squeeze(x, -1))(decoder)

        self._autoencoder = Model(model_inputs, model_outputs)
        self._encoder = Model(model_inputs, encoder)

        decoder_inputs = Input(shape=(latent_dim,))

        self._decoder = Model(decoder_inputs, self._autoencoder.layers[-4](decoder_inputs))

    def get_model(self, lr=0.001):
        optimizer = Adam(lr=lr, epsilon=1e-08, amsgrad=True, clipnorm=1.0)
        self._autoencoder.compile(loss='mse', optimizer=optimizer)
        return self._autoencoder, self._encoder, self._decoder
