from generative_models import utils
from generative_models.wgan.wgan_model import WGAN

dataset, _, timesteps = utils.load_splitted_dataset()
# dataset, _, timesteps = utils.load_resized_mnist()

clip_value = 0.01

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

config_2 = {
    'timesteps': timesteps,
    'run_dir': run_dir,
    'img_dir': img_dir,
    'model_dir': model_dir,
    'generated_datesets_dir': generated_datesets_dir,
    'clip_value': clip_value
}

config = utils.merge_config_and_save(config_2)

wgan = WGAN(config)
losses = wgan.train(dataset)
