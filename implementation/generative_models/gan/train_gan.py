import sys
from keras.callbacks import *

sys.path.append("..")
import utils
from gan import GAN

# normalized_transactions_filepath = "../../../datasets/berka_dataset/usable/normalized_transactions_months.npy"
# timesteps = 90
# dataset = utils.load_splitted_dataset(normalized_transactions_filepath, timesteps)

timesteps = 100
dataset = utils.load_resized_mnist()

np.random.shuffle(dataset)

batch_size = 64
epochs = 1000000
n_discriminator = 1
n_generator = 1
latent_dim = 2
generator_lr = 0.00002
critic_lr = 0.00002
img_frequency = 250
loss_frequency = 500
latent_space_frequency = 1000
model_save_frequency = 25000
dataset_generation_frequency = 25000
dataset_generation_size = 50000
use_mbd = False
use_packing = False
packing_degree = 2

lr_decay_factor = 0.5
lr_decay_steps = 200000

assert (use_mbd and use_packing) is not True
assert (use_packing and packing_degree > 0) or not use_packing

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'timesteps': timesteps,
        'n_discriminator': n_discriminator,
        'n_generator': n_generator,
        'latent_dim': latent_dim,
        'generator_lr': generator_lr,
        'critic_lr': critic_lr,
        'img_frequency': img_frequency,
        'loss_frequency': loss_frequency,
        'latent_space_frequency': latent_space_frequency,
        'model_save_frequency': model_save_frequency,
        'dataset_generation_frequency': dataset_generation_frequency,
        'dataset_generation_size': dataset_generation_size,
        'use_mbd': use_mbd,
        'use_packing': use_packing,
        'packing_degree': packing_degree,
        'run_dir': run_dir,
        'img_dir': img_dir,
        'model_dir': model_dir,
        'generated_datesets_dir': generated_datesets_dir,
        'lr_decay_factor': lr_decay_factor,
        'lr_decay_steps': lr_decay_steps
    }


with open(str(run_dir) + '/config.json', 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

gan = GAN(config)
losses = gan.train(dataset)
