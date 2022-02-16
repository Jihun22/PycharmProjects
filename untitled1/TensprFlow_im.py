#의류 이미지 분류
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

#패션 Mnist 데이터셋 임포트
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images , train_labels) , (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)

print(len(train_labels))
print(train_labels)

print(test_images.shape)
print(len(test_labels))

#데이터 전처리
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images /255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 모델 구성

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#모델 컴파일  손실함수: 훈련중 모델이 얼마나 정확한지 측정  , 옵티마이저: 모델이 인식하는 데이터와 해당 손실 함수를 기반으로 모델이 업데이트 되는방식
#메트릭 : 훈련 및 테스트 단계를 모니터링 하는데 사용
model.compile(optimizer= 'adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#모델 훈련
''' 훈련데이터를 모델에 주입 
모델이 이미지와 레이블을 매핑함
테스트 세트 모델 예측 
예측이 테스트 배열 레이블과 일치하는지 확인 
'''
#모델피드

model.fit(train_images, train_labels, epochs=10)

