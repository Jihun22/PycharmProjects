import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import  tensorflow.datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

#데이터 세트 다운로드
(train_ds ,val_ds ,test_ds) , metadata = tfds.lod(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%]'],
    with_info= True,
    as_supervised=True,
)

#꽃 데이터세트 5개의 클래스
num_classes = metadata.features['label'].num_classes
print(num_classes)

# 데이터세트에서 이미지를 검색 ,사용 , 데이터 증강 수행

get_label_name = metadata.features['label'].int2str

image, label = next(iter(train_ds))
_ = plt.imshow(image)
_ = plt.title(get_label_name(label))
