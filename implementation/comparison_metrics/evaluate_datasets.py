import os
from comparison_metrics.evaluation_framework import EvaluationFramework

timesteps = 90
elements = 10
split = 0.3
iterations_number = 10

transactions_filepath = '../../datasets/berka_dataset/usable/normalized_transactions_months.npy'
models_list = ['svm', 'nn']

base_folder = 'models_comparison'
if not os.path.exists(base_folder):
    os.mkdir(base_folder)

for iteration in range(iterations_number):
    flattening_ranges = [0.0, 0.05, 0.1, 0.15, 0.2]

    for flattening_range in flattening_ranges:
        framework = EvaluationFramework(models_list, transactions_filepath, split, elements, timesteps,
                                        flattening_range,
                                        iteration)

        labels = ['vae', 'wgan_gp', 'wgan_gp_packing', 'wgan_gp_vae']
        if flattening_range == 0.0:
            labels.append('handcrafted')

        title = 'models_comparison'

        base_filepath = 'comparison_datasets/'
        end_filename = 'generated_datasets/1000000_generated_data.npy'
        generated_data_filepaths = []
        for label in labels:
            generated_data_filepaths.append(base_filepath + label + '/' + end_filename)

        framework.set_base_folder(base_folder)
        framework.run_comparison_classification(generated_data_filepaths, labels, title)
        print(flush=True)
