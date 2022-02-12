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

#데이터셋 위치
path = '/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)'
os.listdir(path)
print (path)

train_path = os.path.join(path, "train")
print(os.listdir(train_path))
print("*"*100)
test_path = os.path.join(path, "valid")
print(os.listdir(test_path))

folders = glob('/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/*')
print(folders)

plt.imshow(plt.imread("/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/Tomato___Bacterial_spot/00416648-be6e-4bd4-bc8d-82f43f8a7240___GCREC_Bact.Sp 3110.JPG"))
plt.title("세균반점")
plt.show()

plt.imshow(plt.imread("/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/Tomato___Early_blight/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG"))
plt.title("겹무늬병")
plt.show()

plt.imshow(plt.imread("/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/Tomato___Late_blight/0003faa8-4b27-4c65-bf42-6d9e352ca1a5___RS_Late.B 4946.JPG"))
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

