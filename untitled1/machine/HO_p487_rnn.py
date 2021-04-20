#----------------------------------------------------------------
#  tensorflow로 RNN구현하기
#  page 487
#----------------------------------------------------------------
import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
      tf.reset_default_graph()
      tf.set_random_seed(seed)
      np.random.seed(seed)
reset_graph()

n_inputs = 3
n_neurons = 5

X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell,[X0,X1],
                                                      dtype = tf.float32)
Y0, Y1 = output_seqs

init = tf.global_variables_initializer()

# create some data  (3 inputs)
# Minibatch:         instance 0 instance 1 instance 2  instance 3
X0_batch = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]) # t = 0
X1_batch = np.array([[9, 8, 7], [0, 0, 0], [6, 5, 4], [3, 2, 1]]) # t = 1

with tf.Session() as sess:
      init.run()
      Y0_val, Y1_val = sess.run([Y0, Y1], feed_dict={X0: X0_batch, X1: X1_batch})
      print('\n\nWx = \n{}\n\n'.format(sess.run(Wx)))
      print('\n\nWy = \n{}\n\n'.format(sess.run(Wy)))
      print('\n\nb = \n{}\n\n'.format(sess.run(b)))
print('\n\nY_0 =\n{}\n\n'.format(Y0_val))   # output at time t=0
                # prints instance 0 \\ instance 1 \\ instance 2 \\ instance 3
print('\n\nY_1 =\n{}\n\n'.format(Y1_val))   # output at time t=1
                # prints instance 0 \\ instance 1 \\ instance 2 \\ instance 3


#----------------------------------------------------------------
#  static_rnn을 이용한 Time Step 펼치기
#  page 488
#----------------------------------------------------------------
def reset_graph(seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)

reset_graph()

n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
X_seqs = tf.unstack(tf.transpose(X, perm=[1, 0, 2]))
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units = n_neurons)
output_seqs, states = tf.contrib.rnn.static_rnn(basic_cell, X_seqs, dtype=tf.float32)

outputs = tf.transpose(tf.stack(output_seqs), perm=[1, 0, 2])

X_batch = np.array([
            # t = 0      t = 1
            [[0, 1, 2], [9, 8, 7]], # instance 1
            [[3, 4, 5], [0, 0, 0]], # instance 2
            [[6, 7, 8], [6, 5, 4]], # instance 3
            [[9, 0, 1], [3, 2, 1]], # instance 4
          ])

init = tf.global_variables_initializer()

with tf.Session() as sess:
      init.run()
      outputs_val = outputs.eval(feed_dict={X: X_batch})

print('\n\noutputs_val = \n{}\n'.format(outputs_val))

#----------------------------------------------------------------
#  dynamic_rnn을 이용한 Time Step 펼치기
#  page 491
#----------------------------------------------------------------
import numpy as np
import tensorflow as tf


def reset_graph(seed=42):
      tf.reset_default_graph()
      tf.set_random_seed(seed)
      np.random.seed(seed)

n_steps = 2
n_inputs = 3
n_neurons = 5

reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
init = tf.global_variables_initializer()

X_batch = np.array([
          # t = 0      t = 1
          [[0, 1, 2], [9, 8, 7]], # instance 1
          [[3, 4, 5], [0, 0, 0]], # instance 2
          [[6, 7, 8], [6, 5, 4]], # instance 3
          [[9, 0, 1], [3, 2, 1]], # instance 4
      ])

with tf.Session() as sess:
    init.run()
    outputs_val = outputs.eval(feed_dict={X: X_batch})

print('\n\noutputs_val =\n{}\n\n'.format(outputs_val))
#----------------------------------------------------------------
#  가변길이 입력 시퀀스 다루기
#  page 491
#----------------------------------------------------------------
import numpy as np
import tensorflow as tf


def reset_graph(seed=42):
      tf.reset_default_graph()
      tf.set_random_seed(seed)
      np.random.seed(seed)
reset_graph()

n_steps = 2
n_inputs = 3
n_neurons = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)

# seq_length is a vector indicating the length of the input for each instance
seq_length = tf.placeholder(tf.int32, [None])

# Note sequence_length input
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32,
                                      sequence_length=seq_length)

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
      init.run()
      # Now we need to feed values for both placeholders:
      outputs_val = outputs.eval(feed_dict={X: X_batch, seq_length: seq_length_batch})

print('\n\noutputs_val(Variable Sequences) =\n{}\n\n'.format(outputs_val))

#----------------------------------------------------------------
#  RNN MNIST 훈련하기 I
#  page 495
#----------------------------------------------------------------
import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
  tf.reset_default_graph()
  tf.set_random_seed(seed)
  np.random.seed(seed)
reset_graph()

n_steps = 28  # the image is 28 rows of 28 pixels each
n_inputs = 28
n_neurons = 150
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

n_epochs = 100
batch_size = 150

init = tf.global_variables_initializer()
with tf.Session() as sess:
      init.run()
      for epoch in range(n_epochs):
          for iteration in range(mnist.train.num_examples // batch_size):
              X_batch, y_batch = mnist.train.next_batch(batch_size)
              # print("initial X_batch:", X_batch.shape) (150 x 28 x 28)
              X_batch = X_batch.reshape((-1, n_steps, n_inputs))
              # print("after reshape X_batch: ",X_batch.shape) (150 x 28 x 28)
              sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
          acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
          acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
          print('epoch={} Train accuracy: {} Test accuracy: {}'.format(epoch, acc_train, acc_test))

#----------------------------------------------------------------
#  RNN MNIST 훈련하기 II
#  page 495
#----------------------------------------------------------------
import numpy as np
import tensorflow as tf

def reset_graph(seed=42):
      tf.reset_default_graph()
      tf.set_random_seed(seed)
      np.random.seed(seed)
reset_graph()

n_steps = 28  # the image is 28 rows of 28 pixels each
n_inputs = 28
n_outputs = 10

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

n_neurons = 100
n_layers = 3

# Make three layers, each with 100 neurons
layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,
                                activation=tf.nn.relu)
            for layer in range(n_layers)]
# MultiRNNCell makes the network
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)

outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

states_concat = tf.concat(axis=1, values=states)
logits = tf.layers.dense(states_concat, n_outputs)
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

n_epochs = 10
batch_size = 150

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            X_batch = X_batch.reshape((-1, n_steps, n_inputs))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
    acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print('epoch={}\t` Train accuracy: {}\t Test accuracy: {}'.format(epoch, acc_train, acc_test))
#----------------------------------------------------------------
#  RNN Timeseries
#  page
#----------------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
reset_graph()

t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
      return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
      t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
      Ts = t0 + np.arange(0., n_steps + 1) * resolution
      ys = time_series(Ts)
      return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")

# save_fig("time_series_plot")
plt.show()

X_batch, y_batch = next_batch(1, n_steps)
np.c_[X_batch[0], y_batch[0]]

reset_graph()

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
# OutputProjectionWrapper를 사용한 경우
# cell =tf.contrib.rnn.OutputProjectionWrapper(
#        tf.contrib.rnn.BasicRNNCell(num_units=n_neurons,activation=tf.nn.relu),
#        output_size=n_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

reset_graph()

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

# wrap the cell in an OutputProjectionWrapper
cell = tf.contrib.rnn.OutputProjectionWrapper(
     tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
     output_size=n_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

learning_rate = 0.001

loss = tf.reduce_mean(tf.square(outputs - y)) # MSE
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

saver = tf.train.Saver()

n_iterations = 1500
batch_size = 50

init = tf.global_variables_initializer()
with tf.Session() as sess:
      init.run()
      for iteration in range(n_iterations):
          X_batch, y_batch = next_batch(batch_size, n_steps)
          sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
          if iteration % 100 == 0:
              mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
              print('{}\tMSE = {}'.format(iteration, mse))

      saver.save(sess, "./my_time_series_model") # not shown in the book

with tf.Session() as sess:                          # not shown in the book
      saver.restore(sess, "./my_time_series_model")   # not shown

      X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
      y_pred = sess.run(outputs, feed_dict={X: X_new})

print('\n\ny_pred = \n{}\n\n'.format(y_pred))

plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

# save_fig("time_series_pred_plot")
plt.show()
#----------------------------------------------------------------
#  RNN Timeseries_Dropout
#  page
#----------------------------------------------------------------
import numpy as np
import tensorflow as tf
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

def reset_graph(seed=42):
      tf.reset_default_graph()
      tf.set_random_seed(seed)
      np.random.seed(seed)
reset_graph()


t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
      return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
      t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
      Ts = t0 + np.arange(0., n_steps + 1) * resolution
      ys = time_series(Ts)
      return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("A time series (generated)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
plt.legend(loc="lower left", fontsize=14)
plt.axis([0, 30, -17, 13])
plt.xlabel("Time")
plt.ylabel("Value")

plt.subplot(122)
plt.title("A training instance", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.legend(loc="upper left")
plt.xlabel("Time")

# save_fig("time_series_plot")
plt.show()

X_batch, y_batch = next_batch(1, n_steps)
np.c_[X_batch[0], y_batch[0]]

reset_graph()

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

reset_graph()

reset_graph()

n_inputs = 1
n_neurons = 100
n_layers = 3
n_steps = 20
n_outputs = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

keep_prob = 0.5

cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
            for layer in range(n_layers)]
cells_drop = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob)
            for cell in cells]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells_drop)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

learning_rate = 0.01

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# this code is only usable for training, because the DropoutWrapper class has
# no training parameter, so it always applies dropout, even when the model
# is not being trained, so we must first train the model, then create a
# different model for testing, without the DropoutWrapper.

n_iterations = 1000
batch_size = 50

with tf.Session() as sess:
      init.run()
      for iteration in range(n_iterations):
          X_batch, y_batch = next_batch(batch_size, n_steps)
          _, mse = sess.run([training_op, loss], feed_dict={X: X_batch, y: y_batch})
          if iteration % 100 == 0:
              print('{}\tTraining MSE:{}'.format(iteration, mse))

      saver.save(sess, "./my_dropout_time_series_model")


# now that the model is trained, the model must be created again,
# but without the DropoutWrapper for testing:

reset_graph()

n_inputs = 1
n_neurons = 100
n_layers = 3
n_steps = 20
n_outputs = 1


X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

keep_prob = 0.5

cells = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
         for layer in range(n_layers)]
multi_layer_cell = tf.contrib.rnn.MultiRNNCell(cells)
rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)

learning_rate = 0.01

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
      saver.restore(sess, "./my_dropout_time_series_model")

      X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
      y_pred = sess.run(outputs, feed_dict={X: X_new})

plt.title("Testing the model", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
plt.legend(loc="upper left")
plt.xlabel("Time")

plt.show()

#----------------------------------------------------------------
#  RNN LSTM
#  page
#----------------------------------------------------------------
import numpy as np
import tensorflow as tf

n_steps = 28
n_inputs = 28
n_neurons = 150
n_outputs = 10
n_layers = 3

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")
X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
y_test = mnist.test.labels

def reset_graph(seed=42):
      tf.reset_default_graph()
      tf.set_random_seed(seed)
      np.random.seed(seed)
reset_graph()

learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int32, [None])

lstm_cells = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons)
                for layer in range(n_layers)]
multi_cell = tf.contrib.rnn.MultiRNNCell(lstm_cells)
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)
top_layer_h_state = states[-1][1]
logits = tf.layers.dense(top_layer_h_state, n_outputs, name="softmax")
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

states

top_layer_h_state

n_epochs = 10
batch_size = 150

with tf.Session() as sess:
      init.run()
      for epoch in range(n_epochs):
          for iteration in range(mnist.train.num_examples // batch_size):
              X_batch, y_batch = mnist.train.next_batch(batch_size)
              X_batch = X_batch.reshape((batch_size, n_steps, n_inputs))
              sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
          acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
          acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
          print('Epoch: {}\tTrain accuracy = {}\tTest accuracy = {}'.format(epoch,acc_train, acc_test))
