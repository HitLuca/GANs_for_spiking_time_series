import keras.backend as K
from keras import initializers, regularizers, constraints
from keras.constraints import Constraint
from keras.engine import Layer, InputSpec
import numpy as np
from matplotlib import pyplot as plt


def set_model_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable
    
    
class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches."""

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
#         return K.concatenate([x, minibatch_features], 1)
        return minibatch_features

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
#         return input_shape[0], input_shape[1]+self.nb_kernels
        return input_shape[0], self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
#                   'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def generate_save_images(generator, rows, columns, latent_dim, filenames):
    noise = np.random.normal(0, 1, (rows * columns, latent_dim))
    generated_data = generator.predict(noise)

    plt.subplots(rows, columns, figsize=(15, 5))
    k = 1
    for i in range(rows):
        for j in range(columns):
            plt.subplot(rows, columns, k)
            plt.plot(generated_data[k - 1])
            plt.xticks([])
            plt.yticks([])
            plt.ylim(-1, 1)
            k += 1
    plt.tight_layout()
    for filename in filenames:
        plt.savefig(filename)
    plt.close()


def save_losses(losses, filename):
    plt.figure(figsize=(15, 3))
    plt.plot(losses[0])
    plt.plot(losses[1])
    plt.legend(['generator', 'discriminator'])
    plt.savefig(filename)
    plt.close()


def save_latent_space(latent_dim, generator, filename):
    if latent_dim > 2:
        latent_vector = np.random.normal(0, 1, latent_dim)
    plt.subplots(5, 5, figsize=(15, 5))

    for i, v_i in enumerate(np.linspace(-2, 2, 5, True)):
        for j, v_j in enumerate(np.linspace(-2, 2, 5, True)):
            if latent_dim > 2:
                latent_vector[-2:] = [v_i, v_j]
            else:
                latent_vector = np.array([v_i, v_j])

            plt.subplot(5, 5, i * 5 + j + 1)
            plt.plot(generator.predict(latent_vector.reshape((1, latent_dim))).T)
            plt.xticks([])
            plt.yticks([])
            plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()
    plt.close()