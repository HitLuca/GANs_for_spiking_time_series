import json

from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.models import load_model

from code.LSTM_AutoEncoder.LSTM_AutoEncoder import LSTM_AutoEncoder
from code.LSTM_AutoEncoder.utils import *

balances_filepath = "/home/luca/PycharmProjects/Master-thesis/berka_dataset/parsed/sparse_balances.mtx"
continue_training = False

if continue_training:
    folder_name = '2018-02-13_11-09-11'
    config_filepath = 'logs/' + folder_name + '/config.json'
    config = json.load(open(config_filepath, 'r'))

    batch_num = config['batch_num']
    timesteps = config['timesteps']
    epochs = config['epochs']
    epoch = config['epoch'] + 1

    sequence_autoencoder = load_model('logs/' + folder_name + '/model.h5')
else:
    timesteps = 100
    batch_num = 32
    lstm_size = 256
    latent_dim = 16
    lr = 0.005
    epoch = 0
    epochs = int(1e9)

    sequence_autoencoder = LSTM_AutoEncoder(timesteps, lstm_size, latent_dim).get_model(lr)

    folder_name = create_folders()

    config = {'batch_num': batch_num,
              'timesteps': timesteps,
              'lstm_size': lstm_size,
              'latent_dim': latent_dim,
              'lr': lr,
              'epochs': epochs,
              'epoch': epoch}

    config_filepath = 'logs/' + folder_name + '/config.json'
    json.dump(config, open(config_filepath, 'w'))

balances = get_balances(rescale=True, timesteps=timesteps, balances_filepath=balances_filepath)
N = balances.shape[0]
D = balances.shape[1]
print(N, 'datapoints with', D, 'timesteps')

tensorboard = TensorBoard(log_dir='./logs/' + folder_name, histogram_freq=0, batch_size=32, write_graph=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, min_lr=0.001, verbose=0, cooldown=2)

plots_folder = 'logs/' + folder_name + '/img/'
result_plotter = ResultPlotter(dataset=balances, plots_folder=plots_folder, model=sequence_autoencoder)

model_filepath = 'logs/' + folder_name + '/model.h5'
config_filepath = 'logs/' + folder_name + '/config.json'
model_saver = ModelSaver(model=sequence_autoencoder, model_filepath=model_filepath, config_filepath=config_filepath)

sequence_autoencoder.fit(balances, balances, batch_size=batch_num,
                         epochs=epochs, initial_epoch=epoch,
                         callbacks=[result_plotter, tensorboard, reduce_lr, model_saver])
