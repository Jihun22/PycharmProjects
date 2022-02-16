#의류 이미지 분류
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

#패션 Mnist 데이터셋 임포트
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images , train_labels) , (test_images, test_labels) = fashion_mnist.load_data()

