import sys

from vae import VAE

sys.path.append("..")
import utils

dataset, _, timesteps = utils.load_splitted_dataset()
# dataset, _, timesteps = utils.load_resized_mnist()

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

config_2 = {
    'timesteps': timesteps,
    'run_dir': run_dir,
    'img_dir': img_dir,
    'model_dir': model_dir,
    'generated_datesets_dir': generated_datesets_dir,
}

config = utils.merge_config_and_save(config_2)

vae = VAE(config)
losses = vae.train(dataset)
