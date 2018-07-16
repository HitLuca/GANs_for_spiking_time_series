import sys

sys.path.append("..")
import utils
from ae import AE
from ae_utils import AE_type

dataset, _, timesteps = utils.load_splitted_dataset()
# dataset, _, timesteps = utils.load_resized_mnist()

encoder_type = AE_type.dense
decoder_type = AE_type.dense

run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir()

config_2 = {
    'timesteps': timesteps,
    'encoder_type': encoder_type.name,
    'decoder_type': decoder_type.name,
    'run_dir': run_dir,
    'img_dir': img_dir,
    'model_dir': model_dir,
    'generated_datesets_dir': generated_datesets_dir,
}

config = utils.merge_config_and_save(config_2)

ae = AE(config)
losses = ae.train(dataset)
