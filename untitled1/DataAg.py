import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import  tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#데이터 세트 다운로드
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

#꽃 데이터세트 5개의 클래스
num_classes = metadata.features['label'].num_classes
print(num_classes)

# 데이터세트에서 이미지를 검색 ,사용 , 데이터 증강 수행

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
plt.imshow(image)
plt.title(get_label_name(label))
plt.show()

#크기 및 배율 조정하기

IMG_SIZE =180

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
    layers.experimental.preprocessing.Rescaling(1./255)
])

result = resize_and_rescale(image)
plt.imshow(result)
plt.show()
#픽셀이 [0-1] 있는지 확인
print("Min and max pixel values:", result.numpy().min(), result.numpy().max())


# 데이터 증강 전처리 레이어 사용하여 동일한 이미지를 반복적으로 적용

data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
  layers.experimental.preprocessing.RandomRotation(0.2),
])

# Add the image to a batch
image = tf.expand_dims(image, 0)

plt.figure(figsize=(10, 10))
for i in range(9):
  augmented_image = data_augmentation(image)
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(augmented_image[0])
  plt.axis("off")
  plt.show()

#옵션1. 전처리 레이어를 모델의 일부로 만들기

model = tf.keras.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(16,3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
])

#옵션2. 데이터세트에 전처리 레이어 적용하기

aud_ds =train_ds.map(
    lambda  x, y :(resize_and_rescale(x, training= True), y))

#데이터세트에 전처리 레이어 적용하기

batch_size = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def prepare(ds, shuffle=False, augment=False):
  # Resize and rescale all datasets
  ds = ds.map(lambda x, y: (resize_and_rescale(x), y),
              num_parallel_calls=AUTOTUNE)

  if shuffle:
    ds = ds.shuffle(1000)

  # Batch all datasets
  ds = ds.batch(batch_size)

  # Use data augmentation only on the training set
  if augment:
    ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=AUTOTUNE)

  # Use buffered prefecting on all datasets
  return ds.prefetch(buffer_size=AUTOTUNE)

train_ds = prepare(train_ds, shuffle=True, augment=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)



