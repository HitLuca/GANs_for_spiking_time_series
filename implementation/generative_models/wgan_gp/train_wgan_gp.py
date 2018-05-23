from keras.callbacks import *

from implementation.generative_models import utils
from implementation.generative_models.wgan_gp.wgan_gp import WGAN_GP

normalized_transactions_filepath = "../../../datasets/berka_dataset/usable/normalized_transactions_months.npy"
timesteps = 90
dataset = utils.load_splitted_dataset(normalized_transactions_filepath, timesteps)

# timesteps = 100
# dataset = utils.load_resized_mnist(10)

np.random.shuffle(dataset)

batch_size = 64
epochs = 5000000
n_critic = 5
n_generator = 1
latent_dim = 15
generator_lr = 0.0001
critic_lr = 0.0001
img_frequency = 500
loss_frequency = 500
latent_space_frequency = 1000
model_save_frequency = 25000
dataset_generation_frequency = 25000
dataset_generation_size = 50000
gradient_penality_weight = 10
packing_degree = 1

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

wgan_gp = WGAN_GP(timesteps, latent_dim, packing_degree, gradient_penality_weight, batch_size, run_dir, img_dir, model_dir,
                  generated_datesets_dir)
wgan_gp.build_models(generator_lr, critic_lr)

losses = wgan_gp.train(epochs, n_generator, n_critic, dataset,
                       img_frequency, loss_frequency, latent_space_frequency, model_save_frequency,
                       dataset_generation_frequency, dataset_generation_size)