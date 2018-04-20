from pathlib import Path

import numpy as np
from datetime import datetime

from WGAN import WGAN

normalized_transactions_filepath = "../../datasets/berka_dataset/usable/normalized_transactions_100.npy"

timesteps = 100
transactions = np.load(normalized_transactions_filepath)

batch_size = 64
epochs = 500000
n_critic = 5
n_generator = 1
latent_dim = 2
generator_lr = 0.00005
critic_lr = 0.00005
clip_value = 0.05
img_frequency = 250
model_save_frequency = 3000
dataset_generation_frequency = 25000
dataset_generation_size = 100000

root_path = Path('wgan')
if not root_path.exists():
    root_path.mkdir()

current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

run_dir = root_path / current_datetime
img_dir = run_dir / 'img'
model_dir = run_dir / 'models'
generated_datesets_dir = run_dir / 'generated_datasets'

img_dir.mkdir(parents=True)
model_dir.mkdir(parents=True)
generated_datesets_dir.mkdir(parents=True)

wgan = WGAN(timesteps, latent_dim, run_dir, img_dir, model_dir, generated_datesets_dir)
# gan, generator, critic = wgan.restore_training()
gan, generator, critic = wgan.build_models(generator_lr, critic_lr)

losses = wgan.train(batch_size, epochs, n_generator, n_critic, transactions, clip_value,
                    img_frequency, model_save_frequency, dataset_generation_frequency, dataset_generation_size)