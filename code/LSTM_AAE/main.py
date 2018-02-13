from code.LSTM_AAE.LSTM_AAE import LSTM_AAE
from code.utils import *

balances_filepath = "/home/luca/PycharmProjects/Master-thesis/berka_dataset/parsed/sparse_balances.mtx"

timesteps = 20
batch_size = 128
lstm_size = 256
latent_dim = 6
lr = 0.005
save_interval = 50

epochs = int(1e9)
half_batch = int(batch_size / 2)

lstm_aae, encoder, decoder, discriminator = LSTM_AAE(timesteps, lstm_size, latent_dim).get_model(lr)

folder_name = create_folders()

balances = get_balances(rescale=True, timesteps=timesteps, balances_filepath=balances_filepath)
N = balances.shape[0]
D = balances.shape[1]
print(N, 'datapoints with', D, 'timesteps')

plots_folder = 'logs/' + folder_name + '/img/'
result_plotter = AAEResultPlotter(balances, encoder, decoder, plots_folder)

losses = [[], []]

for epoch in range(epochs):
    # Discriminator training
    idx = np.random.randint(0, N, half_batch)
    chosen_balances = balances[idx]

    latent_fake = encoder.predict(chosen_balances)

    latent_sampled = np.random.normal(size=(half_batch, latent_dim))

    valid = np.ones((half_batch, 1))
    fake = np.zeros((half_batch, 1))

    d_loss_real = discriminator.train_on_batch(latent_sampled, valid)
    d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Generator training
    idx = np.random.randint(0, N, batch_size)
    imgs = balances[idx]

    valid_y = np.ones((batch_size, 1))
    g_loss = lstm_aae.train_on_batch(imgs, [imgs, valid_y])

    # Plot the progress
    print("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (
        epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1]))
    losses[0].append(d_loss[0])
    losses[1].append(g_loss[0])

    if epoch % save_interval == 0:
        result_plotter.plot_results(epoch)
