import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#mnist 데이터셋 다운로드
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

#시각화

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
c = 0
for x in range(5):
    for y in range(3):
        plt.subplot(5, 3, c + 1)
        plt.imshow(x_train[c], cmap='gray')
        c += 1

plt.show()

print(y_train[:15])

# 모델 Sequential 모델 생성
