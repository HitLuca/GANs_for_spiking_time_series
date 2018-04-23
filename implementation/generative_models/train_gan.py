import os
import shutil
import numpy as np

from GAN import GAN

normalized_transactions_filepath = "../../datasets/berka_dataset/usable/normalized_transactions_100.npy"

transactions = np.load(normalized_transactions_filepath)
np.random.shuffle(transactions)
N, D = transactions.shape
print(N, D)

timesteps = 100
batch_size = 64
epochs = int(1e5)
n_discriminator = 1
n_generator = 5
latent_dim = 2
lr = 0.0001
img_frequency = 250
timesteps = timesteps

if os.path.exists('gan'):
    shutil.rmtree('gan')
os.makedirs('gan')

gan = GAN(timesteps, latent_dim)
gan.build_model(lr)
gan.train(batch_size, epochs, n_generator, n_discriminator, transactions, img_frequency)