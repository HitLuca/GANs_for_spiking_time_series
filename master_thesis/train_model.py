import sys

from generative_models import utils
from generative_models.handcrafted.handcrafted_generation import HandcraftedGeneration
from generative_models.vae.vae_model import VAE
from generative_models.wgan.wgan_model import WGAN
from generative_models.wgan_gp.wgan_gp_model import WGAN_GP
from generative_models.wgan_gp_vae.wgan_gp_vae_model import WGAN_GP_VAE

models_dictionary = {
    'handcrafted': HandcraftedGeneration,
    'vae': VAE,
    'wgan': WGAN,
    'wgan_gp': WGAN_GP,
    'wgan_gp_vae': WGAN_GP_VAE
}


def train(model_type):
    dataset, _, timesteps = utils.load_splitted_dataset()
    # dataset, _, timesteps = utils.load_resized_mnist()

    clip_value = 0.01
    use_mbd = False
    use_packing = False
    packing_degree = 2

    run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir(model_type)

    config_2 = {
        'clip_value': clip_value,
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

    model = models_dictionary[model_type](config)
    losses = model.train(dataset)
    return losses


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'vae'
    print(model_type)
    train(model_type)
