import tensorflow as tf
import numpy as np
from datetime import datetime
from code.LSTM_AE.LSTM_AE import LSTM_AE


def generate_simple_dataset(N, D, min_value, max_value):
    dataset = np.zeros((N, D))
    change_probability = 0.01
    for row in range(dataset.shape[0]):
        past_point = np.random.randint(min_value, max_value, 1)
        inclination = np.random.uniform(-10, 10, 1)
        for column in range(dataset.shape[1]):
            new_point = past_point + inclination
            if new_point > max_value:
                new_point = max_value
            elif new_point < min_value:
                new_point = min_value

            past_point = new_point
            dataset[row, column] = new_point
            if np.random.random() < change_probability:
                inclination = np.random.uniform(-10, 10, 1)
    return dataset


lstm_num_hidden = 256
latent_dim = 64
lstm_num_layers = 1
batch_size = 32
learning_rate = 1e-3
gradient_clip = 5

# sparse_transactions = io.mmread("/home/luca/PycharmProjects/Master-thesis/berka_dataset/parsed"
#                                 "/sparse_transactions_min.mtx")
# sparse_transactions = sparse_transactions.todense().astype(float) / np.max(sparse_transactions)

sparse_transactions = generate_simple_dataset(1000, 500, -3000, 3000)
N = sparse_transactions.shape[0]
D = sparse_transactions.shape[1]

sequence_length = D
inputs = tf.placeholder(dtype=tf.float32, shape=[batch_size, sequence_length])

lstm_ae = LSTM_AE(lstm_num_hidden, latent_dim, lstm_num_layers, batch_size, inputs, sequence_length)
loss = lstm_ae.loss()
# train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

tvars = tf.trainable_variables()
grads = tf.gradients(loss, tvars)
grads, _ = tf.clip_by_global_norm(grads, gradient_clip)
numel = tf.constant([[0]])
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients = zip(grads, tvars)
train_step = optimizer.apply_gradients(gradients)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf.summary.scalar('loss', loss)
summaries_op = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs/' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/', sess.graph)

for i in range(100000):
    indexes = np.random.choice(N, batch_size, replace=False)
    minibatch = sparse_transactions[indexes, :]
    result = sess.run([loss, train_step, summaries_op], feed_dict={inputs: minibatch})
    writer.add_summary(result[2], i)
    writer.flush()
    if i % 10 == 0:
        print(result[0])
