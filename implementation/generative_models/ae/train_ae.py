import sys
import json
sys.path.append("..")
import utils
import numpy as np
from ae import AE
from ae_utils import AE_type

# normalized_transactions_filepath = "../../../datasets/berka_dataset/usable/normalized_transactions_months.npy"
# timesteps = 90
# dataset = utils.load_splitted_dataset(normalized_transactions_filepath, timesteps)

timesteps = 100
dataset = utils.load_resized_mnist()

np.random.shuffle(dataset)

encoder_type = AE_type.dense
decoder_type = AE_type.dense

batch_size = 64
epochs = 1000000
latent_dim = 2
lr = 0.001
img_frequency = 5000
loss_frequency = 2500
latent_space_frequency = 5000
model_save_frequency = 100000
dataset_generation_frequency = 100000
dataset_generation_size = 50000

lr_decay_factor = 0.5
lr_decay_steps = 200000

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()


config = {
        'encoder_type': encoder_type.name,
        'decoder_type': decoder_type.name,
        'batch_size': batch_size,
        'epochs': epochs,
        'timesteps': timesteps,
        'latent_dim': latent_dim,
        'lr': lr,
        'img_frequency': img_frequency,
        'loss_frequency': loss_frequency,
        'latent_space_frequency': latent_space_frequency,
        'model_save_frequency': model_save_frequency,
        'dataset_generation_frequency': dataset_generation_frequency,
        'dataset_generation_size': dataset_generation_size,
        'run_dir': run_dir,
        'img_dir': img_dir,
        'model_dir': model_dir,
        'generated_datesets_dir': generated_datesets_dir,
        'lr_decay_factor': lr_decay_factor,
        'lr_decay_steps': lr_decay_steps
    }


with open(str(run_dir) + '/config.json', 'w') as f:
    json.dump(config, f, indent=4, sort_keys=True)

ae = AE(config)
losses = ae.train(dataset)
