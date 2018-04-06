from keras.callbacks import ReduceLROnPlateau, TensorBoard
from keras.models import load_model

from implementation.LSTM_AE.LSTM_AE import LSTM_AE
from implementation.custom_keras_callbacks import AEResultPlotter, AEEmbeddingsVisualizer, AEModelSaver
from implementation.utils import *

balances_filepath = "../../berka_dataset/parsed/balances.npy"

continue_training = True

if continue_training:
    date_time = '2018-02-23_17-36-49'
else:
    date_time = get_date_time()

run_folder = os.path.join('logs', date_time)
config_filepath = os.path.join(run_folder, 'config.json')
plots_folder = os.path.join(run_folder, 'img')

if continue_training:
    timesteps, batch_num, lstm_size, latent_dim, lr, epoch, epochs, print_freq = load_config(config_filepath)
    autoencoder = load_model(os.path.join(run_folder, 'autoencoder.h5'))
    encoder = load_model(os.path.join(run_folder, 'encoder.h5'))
    decoder = load_model(os.path.join(run_folder, 'decoder.h5'))
    epochs += 100
else:
    timesteps = 100
    batch_num = 32
    lstm_size = 256
    latent_dim = 10
    lr = 0.001
    epoch = 0
    epochs = 300  # int(1e9)
    print_freq = 5

    os.makedirs(plots_folder)
    autoencoder, encoder, decoder = LSTM_AE(timesteps, lstm_size, latent_dim).get_model(lr)
    save_config(timesteps, batch_num, lstm_size, latent_dim, lr, epoch, epochs, print_freq, config_filepath)

balances = get_balances(rescale=True, timesteps=timesteps, balances_filepath=balances_filepath)
np.random.shuffle(balances)
balances = balances[:10000]
embeddings_dataset = balances[:5000]
create_dataset_sprite(embeddings_dataset, run_folder)

N, D = balances.shape
print(N, 'datapoints with', D, 'timesteps')

tensorboard = TensorBoard(log_dir=run_folder, histogram_freq=print_freq, write_graph=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.4, patience=5, min_lr=0.0005, verbose=0, cooldown=2)
result_plotter = AEResultPlotter(dataset=balances, plots_folder=plots_folder, model=autoencoder)
model_saver = AEModelSaver(autoencoder=autoencoder, encoder=encoder, decoder=decoder, run_folder=run_folder)
embeddings_visualizer = AEEmbeddingsVisualizer(encoder, embeddings_dataset, run_folder, embeddings_frequency=print_freq)

autoencoder.fit(balances, balances, batch_size=batch_num, validation_split=0.1,
                epochs=epochs, initial_epoch=epoch,
                callbacks=[result_plotter, tensorboard, model_saver, embeddings_visualizer, reduce_lr])
