from tensorflow import keras
from tensorflow.keras.datasets import mnist
# 맷플롯립 라이브러리 사용
import matplotlib.pyplot as plt
#신경망
from  keras import models
from  keras import  layers
from  keras.utils import to_categorical
(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()

print(train_images.shape)

print(len(train_labels))

print(train_labels)

print(test_images.shape)

print(len(test_labels))

print(test_labels)
##### 맷플롯립 라이브러리 사용 ################
digit = train_images[4]

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
#################################

##신경망##
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

#손실함수, 옵티마이저
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 훈련이미지 uint8 타입에서 float32 타입 600000, 28*28 로 변경
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

#레이블 범주형 인코딩
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#fit 메서드 호출하여 훈련 데이터를 모델에 학습
network.fit(train_images , train_labels, epochs=5, batch_size=128)

#훈련 데이터 출력

test_loss, test_acc =network.evaluate(test_images , test_labels)

print('test_acc:',test_acc)
