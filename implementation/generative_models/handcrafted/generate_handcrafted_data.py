import sys
import numpy as np
sys.path.append("..")
import utils
import matplotlib.pyplot as plt


def get_mode(dataset):
    (values, counts) = np.unique(dataset, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]


def calculate_transaction_probability(dataset):
    zero_value = get_mode(dataset)
    dataset[dataset == zero_value] = 0
    dataset[dataset != 0] = 1
    return np.sum(dataset, axis=0)/dataset.shape[0]


dataset, _, timesteps = utils.load_splitted_dataset()
elements = 50000

run_dir, _, _, generated_datesets_dir = utils.generate_run_dir()

dataset_mean = np.mean(dataset, axis=0)
dataset_std = np.std(dataset, axis=0)
dataset_mode = get_mode(dataset)

spike_probability = calculate_transaction_probability(dataset)

sampled = np.ones((elements, timesteps)) * dataset_mode
spikes = np.random.normal(loc=dataset_mean, scale=dataset_std, size=(elements, timesteps))
probabilities = (np.random.random_sample((elements, timesteps)) < spike_probability)
sampled[probabilities] = spikes[probabilities]

print(sampled.shape)

print(generated_datesets_dir)

np.save(generated_datesets_dir + '/1000000_generated_data.npy', sampled)