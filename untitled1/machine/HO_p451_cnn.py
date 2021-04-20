#--------------------------------------------------------------------------
# CNN: 수동 Filter 설정의 예
# page 460, 중국사원 및 꽃 사례
#--------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import load_sample_image

china = load_sample_image('china.jpg')
flower = load_sample_image('flower.jpg')
dataset = np.array([china, flower], dtype=np.float32)

plt.imshow(china)
plt.show()
plt.imshow(flower)
plt.show()

batch_size, height, width, channels = dataset.shape
print('\n\ndataset.shape = {}\n'.format(dataset.shape))
print('\nbatch_size = {}\nheight = {}\nwidth = {}\nchannels = {}\n\n'.format(batch_size, height, width, channels))

# 2개의 Filter 생성
filters = np.zeros(shape=(7,7,channels,2), dtype=np.float32)
filters[:, 3, :, 0] = 1 # 수직선
filters[3, :, :, 1] = 1 # 수평선

# 입력 X와 2개의 필터를 적용한 합성곱의 그래프를 만듦
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
convolution = tf.nn.conv2d(X, filters, strides=[1,2,2,1], padding='SAME')

with tf.Session() as sess:
    output = sess.run(convolution, feed_dict={X: dataset})

plt.imshow(output[0, :, :, 1], cmap='gray')
plt.show()

#--------------------------------------------------------------------------
# CNN: 자동 Filter 설정의 예
# page 460, 중국사원 및 꽃 사례
#--------------------------------------------------------------------------
# 입력 X와 2개의 필터를 적용한 합성곱의 그래프를 만듦

#X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
#conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2,2], padding='SAME')

#--------------------------------------------------------------------------
# Pooling
# page 464
#--------------------------------------------------------------------------
X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
max_pool = tf.nn.max_pool(X, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

with tf.Session() as sess:
    output = sess.run(max_pool, feed_dict={X: dataset})

plt.imshow(output[0].astype(np.uint8))
plt.show()
plt.imshow(output[1].astype(np.uint8))
plt.show()
