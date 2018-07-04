import json

import numpy as np
from matplotlib import pyplot as plt


def combine_data(real_data, fake_data, split):
    N_r = real_data.shape[0]
    N_f = fake_data.shape[0]

    x_train = np.vstack([real_data, fake_data])
    y_train = np.vstack([np.zeros((N_r, 1)), np.ones((N_f, 1))])

    N = x_train.shape[0]
    shuffled_indexes = np.random.permutation(N)

    x_train = x_train[shuffled_indexes]
    y_train = y_train[shuffled_indexes]

    split_index = int(N * split)

    x_test, x_train = x_train[:split_index], x_train[split_index:]
    y_test, y_train = y_train[:split_index], y_train[split_index:]

    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    return (x_train, y_train), (x_test, y_test)


def combine_data_regression(real_data, fake_data, split, targets):
    x_train = np.vstack([real_data, fake_data])

    N = x_train.shape[0]
    shuffled_indexes = np.random.permutation(N)

    x_train = x_train[shuffled_indexes]
    y_train = x_train[:, -targets:]
    x_train = x_train[:, :-targets]

    split_index = int(N * split)

    x_test, x_train = x_train[:split_index], x_train[split_index:]
    y_test, y_train = y_train[:split_index], y_train[split_index:]

    return (x_train, y_train), (x_test, y_test)


def plot_metrics(histories, labels, title, save, save_filename):
    bar_width = 0.2

    runs_names = list(labels)
    models_names = list(histories[runs_names[0]].keys())
    models_names.sort()
    metrics = list(histories[runs_names[0]][models_names[0]].keys())
    metrics.sort()
    n_metrics = len(metrics)

    plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    plt.suptitle(title)

    for index, metric in enumerate(metrics):
        plt.subplot(1, n_metrics, index + 1)

        max_score = 0
        min_score = 1
        for i, model_name in enumerate(models_names):
            model_scores = []
            for run_name in labels:
                score = histories[run_name][model_name][metric]
                max_score = score if score > max_score else max_score
                min_score = score if score < min_score else min_score
                model_scores.append(score)

            axis = np.arange((len(model_scores)))
            plt.bar(axis + bar_width * i, model_scores, width=bar_width, edgecolor='black', label=model_name)
            plt.legend()
        plt.xticks(axis + bar_width * ((len(models_names) -1) / 2), labels, rotation=45)
        plt.ylim(ymin=max(min_score - 0.03, 0), ymax=min(max_score + 0.03, 1))
        plt.ylabel(metric)
    if save:
        plt.savefig(save_filename + '.png')
    else:
        plt.show()


def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def get_mode(dataset):
    (values, counts) = np.unique(dataset, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]
