import tensorflow as tf

tf.reset_default_graph()  # Remove the previous weights and bias

save_file = './model.ckpt'

#weights = tf.Variable(tf.truncated_normal([2, 3]))
#bias = tf.Variable(tf.truncated_normal([3]))
weights = tf.Variable(tf.truncated_normal([2, 3]), name='w1')
bias = tf.Variable(tf.truncated_normal([3]), name='b1')

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('\nWeights = \n{}\nweight_name = {}\n\n'.format(sess.run(weights), weights.name))
    print('\nBias = \n{}\nbias_name = {}\n\n'.format(sess.run(bias), bias.name))
    saver.save(sess, save_file)

tf.reset_default_graph()   # Remove the previous weights and bias

#bias = tf.Variable(tf.truncated_normal([3]))
#weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]), name='b1')
weights = tf.Variable(tf.truncated_normal([2, 3]), name='w1')

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)
    print('\nWeights = \n{}\nRestored weight_name = {}\n\n'.format(sess.run(weights), weights.name))
    print('\nBias = \n{}\nRestored bias_name = {}\n\n'.format(sess.run(bias), bias.name))
