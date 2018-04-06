import keras.backend as K
from keras.layers import BatchNormalization
from keras.layers import Input, Dense, Lambda, Bidirectional, LSTM, \
    RepeatVector, TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam


class LSTM_AAE:
    def __init__(self, timesteps, lstm_size, latent_dim):
        self._timesteps = timesteps
        self._lstm_size = lstm_size
        self._latent_dim = latent_dim

        self._discriminator = self._build_discriminator()
        self._encoder = self._build_encoder()
        self._decoder = self._build_decoder()

    def _build_encoder(self):
        model_inputs = Input(shape=(self._timesteps,))
        expanded_inputs = Lambda(lambda x: K.expand_dims(x, -1))(model_inputs)
        encoded_input = Bidirectional(LSTM(self._lstm_size, return_sequences=False))(expanded_inputs)
        encoded_input = Dense(self._latent_dim, activation='tanh')(encoded_input)
        return Model(model_inputs, encoded_input)

    def _build_decoder(self):
        encoded_input = Input(shape=(self._latent_dim,))
        repeated_inputs = RepeatVector(self._timesteps)(encoded_input)

        decoder_outputs = Bidirectional(LSTM(self._lstm_size, return_sequences=True))(repeated_inputs)
        decoder_outputs = TimeDistributed(Dense(1, activation='tanh'))(decoder_outputs)

        decoded_input = Lambda(lambda x: K.squeeze(x, -1))(decoder_outputs)

        return Model(encoded_input, decoded_input)

    def _build_discriminator(self):
        model = Sequential()

        model.add(Dense(512, input_dim=self._latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1, activation="sigmoid"))

        encoded_repr = Input(shape=(self._latent_dim,))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def get_model(self, lr):
        model_inputs = Input(shape=(self._timesteps,))
        optimizer = Adam(lr=lr, epsilon=1e-08, amsgrad=True, clipnorm=1.0)

        self._discriminator.compile(optimizer=optimizer, loss=['binary_crossentropy'], metrics=['accuracy'])
        self._encoder.compile(optimizer=optimizer, loss=['binary_crossentropy'])
        self._decoder.compile(optimizer=optimizer, loss=['mse'])

        encoded_inputs = self._encoder(model_inputs)
        decoded_inputs = self._decoder(encoded_inputs)

        self._discriminator.trainable = False
        discriminated_encodings = self._discriminator(encoded_inputs)

        LSTM_AAE = Model(model_inputs, [decoded_inputs, discriminated_encodings])
        LSTM_AAE.compile(loss=['mse', 'binary_crossentropy'], loss_weights=[0.999, 0.001], optimizer=optimizer)

        return LSTM_AAE, self._encoder, self._decoder, self._discriminator
