from generative_models import utils
from generative_models.wgan_gp_vae.wgan_gp_vae_model import WGAN_GP_VAE

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

wgan_gp_vae = WGAN_GP_VAE(config)
losses = wgan_gp_vae.train(dataset)
