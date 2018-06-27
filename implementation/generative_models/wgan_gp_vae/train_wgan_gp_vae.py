import sys

from keras.callbacks import *

sys.path.append("..")
import utils
from wgan_gp_vae import WGAN_GP_VAE

normalized_transactions_filepath = "../../../datasets/berka_dataset/usable/normalized_transactions_months.npy"
timesteps = 90
dataset = utils.load_splitted_dataset(normalized_transactions_filepath, timesteps)

# timesteps = 100
# dataset = utils.load_resized_mnist()

np.random.shuffle(dataset)

batch_size = 64
epochs = 1000000
n_critic = 5
n_generator_vae = 1
latent_dim = 2
vae_lr = 0.005
critic_lr = 0.005
img_frequency = 10000
loss_frequency = 5000
latent_space_frequency = 25000
model_save_frequency = 100000
dataset_generation_frequency = 100000
dataset_generation_size = 50000
gradient_penality_weight = 10
gamma = 0.5

lr_decay_factor = 0.5
lr_decay_steps = 200000

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'timesteps': timesteps,
        'n_critic': n_critic,
        'n_generator_vae': n_generator_vae,
        'latent_dim': latent_dim,
        'vae_lr': vae_lr,
        'critic_lr': critic_lr,
        'img_frequency': img_frequency,
        'loss_frequency': loss_frequency,
        'latent_space_frequency': latent_space_frequency,
        'model_save_frequency': model_save_frequency,
        'dataset_generation_frequency': dataset_generation_frequency,
        'dataset_generation_size': dataset_generation_size,
        'gradient_penality_weight': gradient_penality_weight,
        'run_dir': run_dir,
        'img_dir': img_dir,
        'model_dir': model_dir,
        'generated_datesets_dir': generated_datesets_dir,
        'gamma': gamma,
        'lr_decay_factor': lr_decay_factor,
        'lr_decay_steps': lr_decay_steps
    }


with open(str(run_dir) + '/config.json', 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

wgan_gp_vae = WGAN_GP_VAE(config)
losses = wgan_gp_vae.train(dataset)
