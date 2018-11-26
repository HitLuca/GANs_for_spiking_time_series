from generative_models import utils
from generative_models.wgan_gp.wgan_gp_model import WGAN_GP

dataset, _, timesteps = utils.load_splitted_dataset()
# dataset, _, timesteps = utils.load_resized_mnist()

use_mbd = False
use_packing = False
packing_degree = 2

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

config_2 = {
    'timesteps': timesteps,
    'use_mbd': use_mbd,
    'use_packing': use_packing,
    'packing_degree': packing_degree,
    'run_dir': run_dir,
    'img_dir': img_dir,
    'model_dir': model_dir,
    'generated_datesets_dir': generated_datesets_dir,
}

config = utils.merge_config_and_save(config_2)

wgan_gp = WGAN_GP(config)
losses = wgan_gp.train(dataset)
