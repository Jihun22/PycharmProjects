from tensorflow import keras
from tensorflow.keras.datasets import mnist

(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()

print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)