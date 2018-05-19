import keras.backend as K
from keras import Input, Model, Sequential
from keras.layers import *
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import *
from keras.losses import *
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints
from keras.models import load_model
from keras.utils import plot_model

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from datetime import datetime
from pathlib import Path

import pickle
import gc

from time import time
from scipy import signal

transactions_generator_inputs = Input((1,))
transactions_generator_outputs = Dense(100, activation='tanh')(transactions_generator_inputs)
transactions_model = Model(transactions_generator_inputs, transactions_generator_outputs)
transactions_model.compile(loss='mse', optimizer='adam')


generator_inputs = Input(batch_shape=(1, 100,))
latent_space = Dense(50, activation='relu')(generator_inputs)
latent_space = Dense(10, activation='tanh')(latent_space)
latent_space = Dense(50, activation='relu')(latent_space)
kernel_1 = Dense(1*100)(latent_space)
bias_1 = Dense(100)(latent_space)
generator_model = Model(generator_inputs, [kernel_1, bias_1])
generator_model.compile(loss='mse', optimizer='adam')


normalized_transactions_filepath = "../../datasets/berka_dataset/usable/normalized_transactions_100.npy"

timesteps = 100
transactions = np.load(normalized_transactions_filepath)[:50000]
np.random.shuffle(transactions)

training_transactions = transactions[:40000]
test_transactions = transactions[-1000:]


batch_size = 100
fit_epochs_transaction_model = 1000
fit_epochs_generator_model = 1
epochs = 1000000
total_losses = []

early_stopping = EarlyStopping(monitor='loss', min_delta=0.0005, patience=2, verbose=0, mode='auto')

for i in range(1, epochs):
    ones_input = np.ones((batch_size, 1))
    sampled_input = training_transactions[np.random.choice(training_transactions.shape[0])].reshape(1, timesteps)
    repeated_sampled_input = np.tile(sampled_input, (batch_size, 1))

    [kernel_1, bias_1] = generator_model.predict(sampled_input)

    transactions_model.set_weights([kernel_1.reshape(1, 100), 
                                    bias_1.reshape(100)])
    
    transactions_model.fit(ones_input, repeated_sampled_input, epochs=fit_epochs_transaction_model, verbose=0, callbacks=[early_stopping])
    
    model_weights = transactions_model.get_weights()
    generator_model.fit(sampled_input, [model_weights[0].reshape(1, 100), 
                                    model_weights[1].reshape(1, 100)], epochs=fit_epochs_generator_model, verbose=0, callbacks=[early_stopping])
    
    if i%10 == 0:
        losses = []
        for transaction in test_transactions:
            test_input = transaction.reshape(1, timesteps)
            repeated_test_input = np.tile(test_input, (batch_size, 1))

            [kernel_1, bias_1] = generator_model.predict(test_input)

            transactions_model.set_weights([kernel_1.reshape(1, 90), 
                                        bias_1.reshape(90)])

            loss = transactions_model.evaluate(ones_input, repeated_test_input, verbose=0)
            losses.append(loss)
        average_loss = np.mean(np.array(losses))
        total_losses.append(average_loss)

    if i%1000 == 0:
        plots_n = 4
        total_plots = plots_n + 1
        
        test_inputs = test_transactions[np.random.choice(test_transactions.shape[0], plots_n)]
        plt.subplots(1, total_plots, figsize=(15, 3))
        
        for j in range(plots_n):
            plt.subplot(1, total_plots, i+1)

            [kernel_1, bias_1] = generator_model.predict(test_inputs[j:j+1])
            transactions_model.set_weights([kernel_1.reshape(1, 90), 
                                        bias_1.reshape(90)])
            
            plt.plot(test_inputs[j].T)
            plt.plot(transactions_model.predict(ones_input[0:1]).T)
            plt.ylim([-1, 1])
            plt.yticks([])
            plt.xticks([])
        
        plt.subplot(1, total_plots, total_plots)
        plt.suptitle('epoch:' + str(i))
        plt.plot(total_losses)
        plt.tight_layout()
        plt.savefig('gw/' + str(i) + '.png')
        plt.close()
