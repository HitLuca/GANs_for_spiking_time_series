#region imports
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.contrib.tensorboard.plugins import projector
from AE_ts_model import Model, plot_data, plot_z_run
import numpy as np
from scipy import io
import scipy
#endregion imports

"""Hyperparameters"""
config = {}  # Put all configuration information into the dict
config['num_layers'] = 2  # number of layers of stacked RNN's
config['hidden_size'] = 90  # memory cells in a layer
config['max_grad_norm'] = 5  # maximum gradient norm during training
config['batch_size'] = batch_size = 32
config['learning_rate'] = .005
config['crd'] = 1  # Hyperparameter for future generalization
config['num_l'] = 20  # number of units in the latent space

plot_every = 100  # after _plot_every_ GD steps, there's console output
max_iterations = 100000  # maximum number of iterations
dropout = 0.8  # Dropout rate
"""Load the data"""

sparse_transactions = io.mmread("/home/luca/PycharmProjects/Master-thesis/berka_dataset/parsed/sparse_transactions_min.mtx")
sparse_transactions = sparse_transactions.todense()[:10000, :]
N = sparse_transactions.shape[0]
D = sparse_transactions.shape[1]
config['sl'] = sl = D  # sequence length
print('We have %s observations with %s dimensions' % (N, D))

# Proclaim the epochs
epochs = np.floor(batch_size * max_iterations / N)
print('Train with approximately %d epochs' % epochs)

"""Training time!"""
model = Model(config)
sess = tf.Session()

sess.run(model.init_op)
perf_collect = np.zeros((2, int(np.floor(max_iterations / plot_every))))

step = 0  # Step is a counter for filling the numpy array perf_collect
writer = tf.summary.FileWriter('logs/', sess.graph)  # writer for Tensorboard
for i in range(max_iterations):
    batch_ind = np.random.choice(N, batch_size, replace=False)
    result = sess.run([model.loss, model.loss_seq, model.loss_lat_batch, model.train_step, model.merged],
                      feed_dict={model.x: sparse_transactions[batch_ind], model.keep_prob: dropout})

    if i % plot_every == 0:
        # Save train performances
        perf_collect[0, step] = loss_train = result[0]
        loss_train_seq, lost_train_lat = result[1], result[2]

        # and save to Tensorboard
        summary_str = result[3]
        writer.add_summary(summary_str, i)
        writer.flush()

        print("At %6s / %6s train (%5.3f, %5.3f, %5.3f)" % (
            i, max_iterations, loss_train, loss_train_seq, lost_train_lat))
        step += 1