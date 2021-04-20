#-----------------------------------------------
# Saver()를 이용한 tensor 변수값 저장
#-----------------------------------------------
import tensorflow as tf

save_file = './model.ckpt'

weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('\n\nWeight = \n\n{}'.format(sess.run(weights)))
    print('\nbias = \n{}'.format(sess.run(bias)))
    saver.save(sess, save_file)
