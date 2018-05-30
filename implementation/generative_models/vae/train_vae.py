import json
import sys
from vae import VAE
sys.path.append("..")
import utils

# normalized_transactions_filepath = "../../../datasets/berka_dataset/usable/normalized_transactions_months.npy"
# timesteps = 90
# dataset = utils.load_splitted_dataset(normalized_transactions_filepath, timesteps)

timesteps = 100
dataset = utils.load_resized_mnist()

batch_size = 64
epochs = 300000
latent_dim = 15
lr = 0.0001
img_frequency = 1000
loss_frequency = 1000
model_save_frequency = 3000
dataset_generation_frequency = 25000
dataset_generation_size = 100000


run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'timesteps': timesteps,
        'lr': lr,
        'latent_dim': latent_dim,
        'img_frequency': img_frequency,
        'loss_frequency': loss_frequency,
        'model_save_frequency': model_save_frequency,
        'dataset_generation_frequency': dataset_generation_frequency,
        'dataset_generation_size': dataset_generation_size,
        'run_dir': run_dir,
        'img_dir': img_dir,
        'model_dir': model_dir,
        'generated_datesets_dir': generated_datesets_dir
    }


with open(str(run_dir) + '/config.json', 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

vae = VAE(config)

losses = vae.train(dataset)
