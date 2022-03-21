import matplotlib
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg19 import VGG19
import os
import matplotlib.pyplot as plt
import tensorflow as tf

#plt 한글 폰트
matplotlib.rcParams['font.family'] ='Malgun Gothic'

matplotlib.rcParams['axes.unicode_minus'] =False
#plt 한글폰트 끝

#데이터셋 위치 C:\Users\ik533\Desktop\TomatoAug
path = '/Users/ik533/Desktop/TomatoAug'
os.listdir(path)
print (path)

train_path = os.path.join(path, "train")
print(os.listdir(train_path))
print("*"*100)
test_path = os.path.join(path, "valid")
print(os.listdir(test_path))

folders = glob('/Users/ik533/Desktop/TomatoAug/train/*')
print(folders)

plt.imshow(plt.imread("/Users/ik533/Desktop/TomatoAug/train/Tomato___Bacterial_spot/0b13b997-9957-4029-b2a4-ef4a046eb088___UF.GRC_BS_Lab Leaf 0595.JPG"))
plt.title("세균반점")
plt.show()

plt.imshow(plt.imread("/Users/ik533/Desktop/TomatoAug/train/Tomato___Early_blight/0b494c44-8cd0-4491-bdfd-8a354209c3ae___RS_Erly.B 9561.JPG"))
plt.title("겹무늬병")
plt.show()

plt.imshow(plt.imread("/Users/ik533/Desktop/TomatoAug/train/Tomato___Late_blight/0c47de5b-adbe-479f-8ccf-5b8c530c32f8___RS_Late.B 6312.JPG"))
plt.title("감자역병균")
plt.show()
SIZE = [128, 128]
vg19 = VGG19(input_shape=SIZE + [3], weights="imagenet", include_top=False)

for layer in vg19.layers:
    layer.trainable = False

x =Flatten()(vg19.output)

prediction = Dense(len(folders), activation="softmax")(x)
modelvg = Model(inputs=vg19.input , outputs=prediction)

modelvg.summary()

#컴파일 모델
modelvg.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

#데이터 증강
train_datagen_vg19 = ImageDataGenerator(rescale=1./255)

test_datagen_vg19 = ImageDataGenerator(rescale=1./255)

trainning_set_vg19 = train_datagen_vg19.flow_from_directory(train_path,
                                                 target_size=(128, 128),
                                                 batch_size=16,
                                                 class_mode="categorical", shuffle=True)

testing_set_vg19 = test_datagen_vg19.flow_from_directory(test_path,
                                                 target_size=(128, 128),
                                                 batch_size=16,
                                                 class_mode="categorical", shuffle=False)

#모델

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

r_vg19 = modelvg.fit_generator(trainning_set_vg19,
                       validation_data=testing_set_vg19,
                       epochs=2,
                       callbacks=[callback]
                       )

#시각화
accuracy = r_vg19.history['accuracy']
val_accuracy = r_vg19.history['val_accuracy']
loss = r_vg19.history['loss']
val_loss = r_vg19.history['val_loss']
epochs = range(len(accuracy))

plt.title("VGG19")
plt.plot(epochs, accuracy, "b", label="trainning accuracy")
plt.plot(epochs, val_accuracy, "r", label="validation accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, "b", label="trainning loss")
plt.plot(epochs, val_loss, "r", label="validation loss")
plt.legend()
plt.show()
