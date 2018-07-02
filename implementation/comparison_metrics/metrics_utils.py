import json
from matplotlib import pyplot as plt
import numpy as np

def combine_data(real_data, fake_data, split):    
    N_r = real_data.shape[0]
    N_f = fake_data.shape[0]

    x_train = np.vstack([real_data, fake_data]) 
    y_train = np.vstack([np.zeros((N_r, 1)), np.ones((N_f, 1))])

    N = x_train.shape[0]
    shuffled_indexes = np.random.permutation(N)

    x_train = x_train[shuffled_indexes]
    y_train = y_train[shuffled_indexes]

    split_index = int(N*split)

    x_test, x_train = x_train[:split_index], x_train[split_index:]
    y_test, y_train = y_train[:split_index], y_train[split_index:]

    y_train = y_train.reshape(y_train.shape[0])
    y_test = y_test.reshape(y_test.shape[0])

    return (x_train, y_train), (x_test, y_test)

def combine_data_regression(real_data, fake_data, split, targets):    
    N_r = real_data.shape[0]
    N_f = fake_data.shape[0]
    
    x_train = np.vstack([real_data, fake_data]) 

    N = x_train.shape[0]
    shuffled_indexes = np.random.permutation(N)

    x_train = x_train[shuffled_indexes]
    y_train = x_train[:, -targets:]
    x_train = x_train[:, :-targets]

    split_index = int(N*split)

    x_test, x_train = x_train[:split_index], x_train[split_index:]
    y_test, y_train = y_train[:split_index], y_train[split_index:]

    return (x_train, y_train), (x_test, y_test)

def plot_metrics(histories, labels, colors, title): 
    runs_names = list(histories.keys())
    metrics = list(histories[runs_names[0]].keys())
    n_metrics = len(metrics)
    
    plt.subplots(1, n_metrics, figsize=(5*n_metrics, 4))
    plt.suptitle(title)
    
    for index, metric in enumerate(metrics):
        plt.subplot(1, n_metrics, index+1)
        values = []
        for label in labels:
            history = histories[label]
            values.append(history[metric])

        axis = range(len(values))
        plt.bar(axis, values, color=colors)
        plt.xticks(axis, labels, rotation=45)
        plt.ylim(ymin=max(min(values) - 0.03, 0), ymax=min(max(values) + 0.03, 1))
        plt.ylabel(metric)
    plt.show()
    
def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
        
def get_mode(dataset):
    (values,counts) = np.unique(dataset,return_counts=True)
    ind=np.argmax(counts)
    return values[ind]

def postprocess_dataset(dataset, value, step):
    dataset[np.logical_and(dataset < (value + step), dataset > (value - step))] = value
