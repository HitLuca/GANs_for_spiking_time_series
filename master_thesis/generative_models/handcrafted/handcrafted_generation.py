import numpy as np


class HandcraftedGeneration:
    def __init__(self, config):
        self._timesteps = config['timesteps']
        self._dataset_generation_size = config['dataset_generation_size']
        self._generated_datesets_dir = config['generated_datesets_dir']

    @staticmethod
    def _get_mode(dataset):
        (values, counts) = np.unique(dataset, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]

    def _calculate_transaction_probability(self, dataset):
        zero_value = self._get_mode(dataset)
        dataset[dataset == zero_value] = 0
        dataset[dataset != 0] = 1
        return np.sum(dataset, axis=0) / dataset.shape[0]

    def train(self, dataset):
        dataset_mode = self._get_mode(dataset)
        dataset_mean_positive = np.mean(dataset[dataset > dataset_mode], axis=0)
        dataset_std_positive = np.std(dataset[dataset > dataset_mode], axis=0)

        dataset_mean_negative = np.mean(dataset[dataset < dataset_mode], axis=0)
        dataset_std_negative = np.std(dataset[dataset < dataset_mode], axis=0)

        spike_probability = self._calculate_transaction_probability(dataset)

        sampled = np.ones((self._dataset_generation_size, self._timesteps)) * dataset_mode
        spikes_positive = np.random.normal(loc=dataset_mean_positive, scale=dataset_std_positive, size=(self._dataset_generation_size, self._timesteps))
        spikes_negative = np.random.normal(loc=dataset_mean_negative, scale=dataset_std_negative, size=(self._dataset_generation_size, self._timesteps))

        probabilities_positive = (np.random.random_sample((self._dataset_generation_size, self._timesteps)) < spike_probability / 2)
        probabilities_negative = (np.random.random_sample((self._dataset_generation_size, self._timesteps)) < spike_probability / 2)
        sampled[probabilities_positive] = spikes_positive[probabilities_positive]
        sampled[probabilities_negative] = spikes_negative[probabilities_negative]

        np.save(self._generated_datesets_dir + '/1000000_generated_data.npy', sampled)
