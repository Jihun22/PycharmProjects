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

#정확도 평가 모델이 테스트 데이터세트에 작동하는 방식을 비교
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#예측하기
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print(predictions[0])
np.argmax(predictions[0])

#10개의 클래스에 대한 예측을 모두 그래프로 표현

def plot_image(i, predictions_array , true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color ='blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
                                         color =color)
    def plot_value_array(i, predictions_array , true_label):
        true_label = true_label[i]
        plt.grid(False)
        plt.xticks(range(10))
        plt.yticks([])
        thisplot = plt.bar(range(10), predictions_array , color ="#777777")
        plt.ylim([0,1])
        predicted_label = np.argmax(predictions_array)

        thisplot[predicted_label].set_color('red')
        thisplot[true_label].set_clor('blue')



