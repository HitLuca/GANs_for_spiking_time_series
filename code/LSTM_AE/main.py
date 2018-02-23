from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.models import load_model

from code.LSTM_AE.LSTM_AE import LSTM_AE
from code.utils import *


balances_filepath = "../../berka_dataset/parsed/balances.npy"

continue_training = False

if continue_training:
    date_time = '2018-02-14_21-01-37'
else:
    date_time = get_date_time()


run_folder = os.path.join('logs', date_time)
os.mkdir(run_folder)

config_filepath = os.path.join(run_folder, 'config.json')
plots_folder = os.path.join(run_folder, 'img')
os.mkdir(plots_folder)


if continue_training:
    config = json.load(open(config_filepath, 'r'))

    batch_num = config['batch_num']
    timesteps = config['timesteps']
    epochs = config['epochs']
    epoch = config['epoch'] + 1

    autoencoder, encoder, decoder = load_model(os.path.join(run_folder, '/model.h5'))
else:
    timesteps = 100
    batch_num = 32
    lstm_size = 256
    latent_dim = 10
    lr = 0.001
    epoch = 0
    epochs = int(1e9)

    autoencoder, encoder, decoder = LSTM_AE(timesteps, lstm_size, latent_dim).get_model(lr)

    config = {'batch_num': batch_num,
              'timesteps': timesteps,
              'lstm_size': lstm_size,
              'latent_dim': latent_dim,
              'lr': lr,
              'epochs': epochs,
              'epoch': epoch}

    json.dump(config, open(config_filepath, 'w'))

balances = get_balances(rescale=True, timesteps=timesteps, balances_filepath=balances_filepath)
np.random.shuffle(balances)

embeddings_dataset = balances[:5000]
create_dataset_sprite(embeddings_dataset, run_folder)

N = balances.shape[0]
D = balances.shape[1]
print(N, 'datapoints with', D, 'timesteps')

tensorboard = TensorBoard(log_dir=run_folder, histogram_freq=1, write_graph=True)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, min_lr=0.001, verbose=0, cooldown=2)
result_plotter = AEResultPlotter(dataset=balances, plots_folder=plots_folder, model=autoencoder)
model_saver = ModelSaver(model=autoencoder, run_folder=run_folder)
embeddings_visualizer = EmbeddingsVisualizer(encoder, embeddings_dataset, run_folder)

autoencoder.fit(balances, balances, batch_size=batch_num, validation_split=0.2,
                     epochs=epochs, initial_epoch=epoch,
                     callbacks=[result_plotter, tensorboard, model_saver, embeddings_visualizer])

