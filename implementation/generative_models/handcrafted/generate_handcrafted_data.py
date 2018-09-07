import sys
import numpy as np

sys.path.append("..")
import utils


def get_mode(dataset):
    (values, counts) = np.unique(dataset, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]


def calculate_transaction_probability(dataset):
    zero_value = get_mode(dataset)
    dataset[dataset == zero_value] = 0
    dataset[dataset != 0] = 1
    return np.sum(dataset, axis=0) / dataset.shape[0]


dataset, _, timesteps = utils.load_splitted_dataset()
elements = 50000

run_dir, _, _, generated_datesets_dir = utils.generate_run_dir()

dataset_mode = get_mode(dataset)
dataset_mean_positive = np.mean(dataset[dataset > dataset_mode], axis=0)
dataset_std_positive = np.std(dataset[dataset > dataset_mode], axis=0)

dataset_mean_negative = np.mean(dataset[dataset < dataset_mode], axis=0)
dataset_std_negative = np.std(dataset[dataset < dataset_mode], axis=0)

spike_probability = calculate_transaction_probability(dataset)

sampled = np.ones((elements, timesteps)) * dataset_mode
spikes_positive = np.random.normal(loc=dataset_mean_positive, scale=dataset_std_positive, size=(elements, timesteps))
spikes_negative = np.random.normal(loc=dataset_mean_negative, scale=dataset_std_negative, size=(elements, timesteps))

probabilities_positive = (np.random.random_sample((elements, timesteps)) < spike_probability/2)
probabilities_negative = (np.random.random_sample((elements, timesteps)) < spike_probability/2)
sampled[probabilities_positive] = spikes_positive[probabilities_positive]
sampled[probabilities_negative] = spikes_negative[probabilities_negative]

print(sampled.shape)

print(generated_datesets_dir)

np.save(generated_datesets_dir + '/1000000_generated_data.npy', sampled)
