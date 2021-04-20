import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
      tf.reset_default_graph()
      tf.set_random_seed(seed)
      np.random.seed(seed)

reset_graph()

n_steps, n_inputs, n_neurons = 2,  3,  5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

# seq_length is a vector indicating the length of the input for each instance
seq_length = tf.placeholder(tf.int32, [None])

# Note sequence_length input
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,sequence_length=seq_length)
X_batch = np.array([
          # t = 0      t = 1
          [[0, 1, 2], [9, 8, 7]], # instance 1
          [[3, 4, 5], [0, 0, 0]], # instance 2 (1 time step; padded with 0 vector
          [[6, 7, 8], [6, 5, 4]], # instance 3
          [[9, 0, 1], [3, 2, 1]], # instance 4
      ])
seq_length_batch = np.array([2, 1, 2, 2]) # lengths.  Note second is length 1

init = tf.global_variables_initializer()
with tf.Session() as sess:
      sess.run(init)
      # Now we need to feed values for both placeholders:
      outputs_val, states_val = sess.run([outputs, states], feed_dict={X: X_batch,seq_length: seq_length_batch})
print('\n\noutput(dynamic_rnn) =\n{}'.format(outputs_val))
