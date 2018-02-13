import keras
import keras.backend as K
from keras import Input, Model
from keras.layers import Lambda, LSTM, RepeatVector, Dense, TimeDistributed, Bidirectional


class LSTM_AutoEncoder:
    def __init__(self, timesteps, lstm_size, latent_dim):
        model_inputs = Input(shape=(timesteps,))
        expanded_inputs = Lambda(lambda x: K.expand_dims(x, -1))(model_inputs)
        encoded_input = Bidirectional(LSTM(lstm_size, return_sequences=False))(expanded_inputs)
        encoded_input = Dense(latent_dim, activation='tanh')(encoded_input)

        repeated_inputs = RepeatVector(timesteps)(encoded_input)

        decoder_outputs = Bidirectional(LSTM(lstm_size, return_sequences=True))(repeated_inputs)
        decoder_outputs = TimeDistributed(Dense(1, activation='tanh'))(decoder_outputs)

        model_outputs = Lambda(lambda x: K.squeeze(x, -1))(decoder_outputs)

        self._sequence_autoencoder = Model(model_inputs, model_outputs)

    def get_model(self, lr):
        optimizer = keras.optimizers.Adam(lr=lr, epsilon=1e-08, amsgrad=True, clipnorm=1.0)
        self._sequence_autoencoder.compile(loss='mse', optimizer=optimizer)
        return self._sequence_autoencoder
