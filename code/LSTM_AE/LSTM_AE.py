import tensorflow as tf


class LSTM_AE:
    def __init__(self, lstm_num_hidden, latent_dim, lstm_num_layers, batch_size, inputs, seq_length):
        self._lstm_num_hidden = lstm_num_hidden
        self._latent_dim = latent_dim
        self._lstm_num_layers = lstm_num_layers
        self._batch_size = batch_size
        self._inputs = inputs
        self._seq_length = seq_length
        self.build_model()

    def build_model(self):
        expanded_inputs = tf.expand_dims(self._inputs, -1)

        with tf.variable_scope('encoder'):
            encoder = tf.contrib.rnn.MultiRNNCell(
                [self._lstm_cell(self._lstm_num_hidden) for _ in range(self._lstm_num_layers)])

            encoder_outputs, _ = tf.nn.dynamic_rnn(cell=encoder,
                                                   inputs=expanded_inputs,
                                                   dtype=tf.float32)

            cell_output = encoder_outputs[:, -1, :]

        with tf.variable_scope('encoder_to_latent'):
            self._z_mu = tf.layers.dense(cell_output, self._latent_dim)
            self._z_mean, self._z_var = tf.nn.moments(self._z_mu, axes=[1])

        with tf.variable_scope('latent_to_decoder'):
            z_state = tf.layers.dense(self._z_mu, self._lstm_num_hidden)

        with tf.variable_scope('decoder'):
            decoder = tf.contrib.rnn.MultiRNNCell(
                [self._lstm_cell(self._lstm_num_hidden) for _ in range(self._lstm_num_layers)])

            initial_state_dec = tuple([tf.nn.rnn_cell.LSTMStateTuple(z_state, z_state)] * self._lstm_num_layers)
            decoder_inputs = tf.zeros((self._batch_size, self._seq_length, 1))

            decoder_outputs, _ = tf.nn.dynamic_rnn(cell=decoder,
                                          inputs=decoder_inputs,
                                          initial_state=initial_state_dec)

        with tf.variable_scope('decoder_to_output'):
            outputs = tf.layers.dense(decoder_outputs, 1)
            self._outputs = tf.squeeze(outputs, axis=-1)

        return self._outputs

    def loss(self):
        loss = tf.reduce_mean(tf.squared_difference(self._inputs, self._outputs))
        latent_space_loss = tf.reduce_mean(tf.square(self._z_mean) + self._z_var - tf.log(self._z_var + 1e-9) - 1)
        return loss + latent_space_loss

    @staticmethod
    def _lstm_cell(lstm_num_hidden):
        cell = tf.contrib.rnn.BasicLSTMCell(lstm_num_hidden)
        cell = tf.contrib.rnn.DropoutWrapper(cell)
        return cell
