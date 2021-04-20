#-----------------------------------------------
# restore()를 이용한 tensor 변수값 호출
#-----------------------------------------------
import tensorflow as tf

tf.reset_default_graph()  # remove previous weights and bias
save_file = './model.ckpt'

weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, save_file)
    print('\n\nRestored Weight = \n\n{}'.format(sess.run(weights)))
    print('\nRestored bias = \n{}'.format(sess.run(bias)))
