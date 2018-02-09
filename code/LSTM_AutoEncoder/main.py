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

from LSTM_AutoEncoder import LSTM_AutoEncoder

config = tf.ConfigProto(device_count={'GPU': 1})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


class ResultPlotter(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        plt.subplots(2, 2, figsize=(10, 3))
        indexes = range(4)
        for i in range(4):
            plt.subplot(2, 2, i + 1)
            plt.plot(sparse_balances[indexes[i], -timesteps:])
            result = sequence_autoencoder.predict([np.reshape(sparse_balances[indexes[i], -timesteps:], (1, timesteps)), np.zeros((datapoints))])
            plt.plot(result.T)
            plt.xticks([])
            plt.ylim(0, 1)
            plt.yticks([])
        plt.tight_layout()
        plt.savefig('logs/' + folder_name + '/img/' + str(epoch) + '.png')
        plt.close()
        return


folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
os.makedirs('logs/' + folder_name + '/img/')

sparse_balances = io.mmread("/home/luca/PycharmProjects/Master-thesis/berka_dataset/parsed/sparse_balances.mtx")
sparse_balances = sparse_balances.todense()
scaler = MinMaxScaler(feature_range=(0, 1))
sparse_balances = scaler.fit_transform(sparse_balances)

N = sparse_balances.shape[0]
D = sparse_balances.shape[1]

batch_num = 32
timesteps = D
latent_dim = 256
datapoints = N
lr = 0.001
reverse = True

sequence_autoencoder = SequenceAutoencoder(timesteps, latent_dim, reverse).get_model(lr)

# earlyStopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/' + folder_name, histogram_freq=0, batch_size=32, write_graph=True)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, min_lr=0.005, verbose=1)
result_plotter = ResultPlotter()

inputs = [sparse_balances[:datapoints, -timesteps:], np.zeros((datapoints))]
outputs = sparse_balances[:datapoints, -timesteps:]

sequence_autoencoder.fit(inputs, outputs,
                         batch_size=batch_num, epochs=1000, callbacks=[result_plotter, tensorboard])
