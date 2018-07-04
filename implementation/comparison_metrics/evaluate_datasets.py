from evaluator import Evaluator
import sys
sys.path.append("..")
from generative_models import utils
import os

timesteps = 90
elements = 20
split = 0.3
targets = 20

transactions_filepath = '../../datasets/berka_dataset/usable/normalized_transactions_months.npy'
models_list = ['nn', 'rf', 'svm', 'dt']


evaluator = Evaluator(models_list, transactions_filepath, split, elements, timesteps, targets)


# Models Comparison
labels = ['vanilla', 'mbd', 'packing']
base_folder = 'models_comparison'

if not os.path.exists(base_folder):
    os.mkdir(base_folder)

title = 'models_comparison'

base_filepath = '../generative_models/wgan_gp/outputs/final/'
end_filename = 'generated_datasets/1000000_generated_data.npy'
generated_data_filepaths = []
for label in labels:
    generated_data_filepaths.append(base_filepath + label + '/' + end_filename)

evaluator.set_base_folder(base_folder)
evaluator.run_comparison(generated_data_filepaths, labels)
evaluator.save_histories(title)
evaluator.save_metrics_plot(labels, title)


labels = ['100k', '200k', '300k', '400k', '500k', '600k', '700k', '800k', '900k', '1M']

# Vanilla training
base_folder = 'vanilla_training'

if not os.path.exists(base_folder):
    os.mkdir(base_folder)

title = 'vanilla_training'

base_filepath = '../generative_models/wgan_gp/outputs/final/vanilla/generated_datasets/'
generated_data_filepaths = []
for n in range(100000, 1000001, 100000):
    generated_data_filepaths.append(base_filepath + str(n) + '_generated_data.npy')

evaluator.set_base_folder(base_folder)
evaluator.run_comparison(generated_data_filepaths, labels)
evaluator.save_histories(title)
evaluator.save_metrics_plot(labels, title)
