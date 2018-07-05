import os
from datetime import datetime

import keras
import keras.backend as K
import numpy as np
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.misc import imresize


def set_model_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable


def save_samples(generated_data, rows, columns, filenames):
    is_image = False
    if generated_data.shape[1] == 100:
        is_image = True

    if is_image:
        plt.subplots(rows, columns, figsize=(7, 7))
    else:
        plt.subplots(rows, columns, figsize=(columns * 3, rows))

    k = 1
    for i in range(rows):
        for j in range(columns):
            plt.subplot(rows, columns, k)
            if is_image:
                plt.imshow((generated_data[k - 1].reshape(10, 10) + 1.0) / 2.0)
            else:
                plt.plot(generated_data[k - 1].T)
                plt.ylim(-1, 1)
            plt.xticks([])
            plt.yticks([])
            k += 1
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for filename in filenames:
        plt.savefig(filename)
    plt.clf()
    plt.close()


def save_losses_wgan(losses, filename, legend_name='critic'):
    plt.figure(figsize=(15, 4.5))
    plt.plot(losses[0])
    plt.plot(losses[1])
    plt.legend(['generator', legend_name])
    plt.savefig(filename)
    plt.clf()
    plt.close()


def save_losses_other(losses, filename, text):
    plt.figure(figsize=(15, 4.5))
    plt.plot(losses)
    plt.legend([text])
    plt.savefig(filename)
    plt.clf()
    plt.close()


def save_losses_wgan_gp_ae(losses, filename, legend_name='generator AE'):
    plt.subplots(2, 1, figsize=(15, 9))
    plt.subplot(2, 1, 1)
    plt.plot(losses[0])
    plt.plot(losses[1])
    plt.legend(['generator', 'critic'])

    plt.subplot(2, 1, 2)
    plt.plot(losses[2])
    plt.legend([legend_name])

    plt.savefig(filename)
    plt.clf()
    plt.close()


def save_latent_space(generated_data, grid_size, filenames):
    is_image = False
    if generated_data.shape[1] == 100:
        is_image = True

    if is_image:
        plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    else:
        plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
            if is_image:
                plt.imshow((generated_data[i * grid_size + j].reshape(10, 10) + 1.0) / 2.0)
            else:
                plt.plot((generated_data[i * grid_size + j]).T)
                plt.ylim(-1, 1)
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for filename in filenames:
        plt.savefig(filename)
    plt.clf()
    plt.close()


def split_data(dataset, timesteps):
    D = dataset.shape[1]
    if D < timesteps:
        return None
    elif D == timesteps:
        return dataset
    else:
        splitted_data, remaining_data = np.hsplit(dataset, [timesteps])
        remaining_data = split_data(remaining_data, timesteps)
        if remaining_data is not None:
            return np.vstack([splitted_data, remaining_data])
        return splitted_data


def load_splitted_dataset(filepath, timesteps):
    dataset = np.load(filepath)
    dataset = split_data(dataset, timesteps)
    return dataset


def load_resized_mnist():
    from keras.datasets import mnist
    (x_train, y_train), _ = mnist.load_data()
    dataset = np.empty((60000, 10, 10))
    for row in range(x_train.shape[0]):
        dataset[row] = imresize(x_train[row], (10, 10))
    dataset = (dataset / 255.0) * 2.0 - 1.0
    dataset = dataset.reshape(60000, 10 * 10)
    return dataset


class MinibatchDiscrimination(Layer):
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

        self.b = self.add_weight(shape=(self.nb_kernels,),
                                 initializer=keras.initializers.zeros(),
                                 name='kernel',
                                 regularizer=self.W_regularizer,
                                 trainable=True,
                                 constraint=self.W_constraint)

        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        minibatch_features += self.b
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1] + self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def generate_run_dir():
    root_path = 'outputs'
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run_dir = root_path + '/' + current_datetime
    img_dir = run_dir + '/img'
    model_dir = run_dir + '/models'
    generated_datesets_dir = run_dir + '/generated_datasets'

    os.mkdir(run_dir)
    os.mkdir(img_dir)
    os.mkdir(model_dir)
    os.mkdir(generated_datesets_dir)

    return run_dir, img_dir, model_dir, generated_datesets_dir


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class BatchNormalizationGAN(Layer):
    def __init__(self, axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True,
                 beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',
                 beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None, **kwargs):
        super(BatchNormalizationGAN, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=True)  # THIS IS THE PROBLEM, changed training=training into training=True

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(BatchNormalizationGAN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
