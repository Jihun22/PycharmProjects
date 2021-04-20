#-----------------------------------------------
# Saver()를 이용한 Model 불러오기
#-----------------------------------------------
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

mnist = input_data.read_data_sets('.', one_hot=True)

features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), bias)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

save_file = './model.ckpt'
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    saver.restore(sess, save_file)
    test_accuracy = sess.run(accuracy, feed_dict={features: mnist.test.images, labels: mnist.test.labels})
    print('\n\nTest Accuracy = {}\n\n'.format(test_accuracy))
