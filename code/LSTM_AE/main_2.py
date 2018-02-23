from keras import Input, Model
from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, LSTM, Bidirectional, Lambda, TimeDistributed
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K
from tensorflow.contrib.tensorboard.plugins import projector

from code.LSTM_AE.LSTM_AE import LSTM_AE
from code.utils import *
import tensorflow as tf

balances_filepath = "../../berka_dataset/parsed/balances.npy"

timesteps = 81
lstm_size = 128
latent_dim = 5
N = 100
dim = int(np.sqrt(timesteps))

# os.mkdir('logs/test')

balances = get_balances(balances_filepath, timesteps=timesteps)
np.random.shuffle(balances)
balances = balances[:N]

create_dataset_sprite(balances, 'logs/test/sprite.png')
autoencoder, encoder, decoder = LSTM_AE(timesteps, lstm_size, latent_dim).get_model(0.01)

model_saver = ModelSaver(autoencoder, 'logs/test/model.h5', 'logs/test')

autoencoder.fit(balances, balances, epochs=2, callbacks=[model_saver])

embedded_data = encoder.predict(balances)

visualize_embeddings(embedded_data, 'logs/test', 'logs/test/sprite.png', dim, 'logs/test/model.ckpt')