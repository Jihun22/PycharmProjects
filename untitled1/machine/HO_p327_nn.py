#--------------------------------------------------------------------------------
#  iris dataset에 대한 perceptron분석
# page 335
#--------------------------------------------------------------------------------
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
import os

iris = load_iris()
X = iris.data[:, (2,3)] # 꽃잎의 길이와 너비
y = (iris.target == 0).astype(np.int)  #iris Setosa인가?

per_clf = Perceptron(random_state=42)
per_clf.fit(X, y)

y_pred = per_clf.predict([[2, 0.5]])

print('\ny_perd[2.0, 0.5] = {}\n\n'.format(y_pred))

os.system('Pause')
#--------------------------------------------------------------------------------
#  Activation functions
#--------------------------------------------------------------------------------
# Step Functions
def step_function(x):
    y = x>0
    return y.astype(np.int)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def leakyrelu_func(x): # Leaky ReLU(Rectified Linear Unit, 정류된 선형 유닛) 함수
    return (x>=0)*x + (x<0)*0.01*x
def tanh_func(x):
    return np.tanh(x)

import numpy as np
import matplotlib.pylab as plt

x = np.arange(-5.0, 5.0, 0.1)
step = step_function(x)
sigmoid = sigmoid(x)
ReLU = relu(x)
Leak_ReLU = leakyrelu_func(x)
Hyper_tangent = tanh_func(x)
plt.plot(x,step)
plt.ylim(-.1,1.3)
plt.show()
os.system('Pause')

plt.plot(x,sigmoid)
plt.ylim(-.1,1.3)
plt.show()
os.system('Pause')

plt.plot(x,ReLU)
plt.ylim(-.1,2.0)
plt.show()
os.system('Pause')

#plt.plot(x,Leak_ReLU)
plt.plot(x,Hyper_tangent)
plt.ylim(-1.2,1.2)
plt.show()
os.system('Pause')

#--------------------------------------------------------------------------------
#  MLP 구현
#--------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mat
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from sklearn.tree import export_graphviz
import graphviz

mglearn.plots.plot_logistic_regression_graph().view('c:/workspace/t_g1.pdf')
os.system('Pause')
mglearn.plots.plot_single_hidden_layer_graph().view('c:/workspace/t_g2.pdf')
os.system('Pause')
mglearn.plots.plot_two_hidden_layer_graph().view('c:/workspace/t_g3.pdf')
os.system('Pause')

# Activation Function
line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label='tanh')
plt.plot(line, np.maximum(line, 0), label='relu')
plt.legend(loc='best')

plt.xlabel('X')
plt.ylabel('relu(X), tanh(X)')
plt.show()

#----------------------------------------------------------------------
# DNN
# page 340
#----------------------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/')
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype('int')
y_test = mnist.test.labels.astype('int')

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[300, 100], n_classes=10,
                                         feature_columns=feature_cols)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)

#----------------------------------------------------------------------
# DNN
# page 340
#----------------------------------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/')
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype('int')
y_test = mnist.test.labels.astype('int')

feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf = tf.contrib.learn.DNNClassifier(hidden_units=[100, 100], n_classes=10,
                                         feature_columns=feature_cols)
dnn_clf = tf.contrib.learn.SKCompat(dnn_clf)
dnn_clf.fit(X_train, y_train, batch_size=50, steps=40000)

from sklearn.metrics import accuracy_score
y_pred = dnn_clf.predict(X_test)
print('\naccuracy_score = {}\n\n'.format(
                           accuracy_score(y_test, y_pred['classes'])))

#----------------------------------------------------------------------
# Low Level API를 사용한 DNN
# page 341
#----------------------------------------------------------------------
import tensorflow as tf
import numpy as np
import os

n_inputs = 28*28
n_hidden1 = 300
n_hidden2 = 200
n_outputs = 10

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
y = tf.placeholder(tf.int64, shape=(None), name='y')

def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs+n_neurons)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name='kernel')
        b = tf.Variable(tf.zeros([n_neurons]), name='bias')
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z

with tf.name_scope('dnn'):
    hidden1 = neuron_layer(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
    hidden2 = neuron_layer(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
    logits = neuron_layer(hidden2, n_outputs, name='outputs')

#with tf.name_scope('dnn'):
    #hidden1 = tf.layer.dense(X, n_hidden1, name='hidden1', activation=tf.nn.relu)
    #hidden2 = tf.layer.dense(hidden1, n_hidden2, name='hidden2', activation=tf.nn.relu)
    #logits = tf.layer.dense(hidden2, n_outputs, name='outputs')

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                  labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name='loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('evel'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/')


n_epochs = 20
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print('{}  Train accuracy: {:.5f}  Test accuracy: {:.5f}'.format(epoch, acc_train, acc_test))

    save_path = saver.save(sess, "./my_model_final.ckpt")

os.system('Pause')

#----------------------------------------------------------------------
# 신경망 이용하기
# page 347
#----------------------------------------------------------------------

with tf.Session() as sess:
    saver.restore(sess, save_path) #"my_model_final.ckpt")
    X_new_scaled = mnist.test.images[:20]
    Z = logits.eval(feed_dict={X: X_new_scaled})
    print('argmax(Z)              = {}'.format(np.argmax(Z, axis=1)))
    print('mnist.test.labels[:20] = {}'.format(mnist.test.labels[:20]))

os.system('Pause')

# Use Fully connected

tf.reset_default_graph()

from tensorflow.contrib.layers import fully_connected

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")

with tf.name_scope("dnn"):
     hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
     hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
     logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope="outputs")

with tf.name_scope("loss"):
     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
     loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("train"):
     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
     training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
     correct = tf.nn.in_top_k(logits, y, 1)
     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_epochs = 20
n_batches = 50

with tf.Session() as sess:
     init.run()
     for epoch in range(n_epochs):
         for iteration in range(mnist.train.num_examples // batch_size):
             X_batch, y_batch = mnist.train.next_batch(batch_size)
             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
         acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
         acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
         print('{}  Train accuracy: {:.5f}  Test accuracy: {:.5f}'.format(epoch, acc_train, acc_test))
     save_path = saver.save(sess, "./my_model_final.ckpt")
