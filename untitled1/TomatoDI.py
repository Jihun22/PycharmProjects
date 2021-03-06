import matplotlib
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
#plt 한글 폰트
matplotlib.rcParams['font.family'] ='Malgun Gothic'

matplotlib.rcParams['axes.unicode_minus'] =False
#plt 한글폰트 끝
image_size = [224, 224]

vgg = VGG16(input_shape = image_size + [3], weights = 'imagenet', include_top =  False)

for layer in vgg.layers:
    layer.trainable = False

from glob import glob
#데이터셋 위치
folders = glob('/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/*')

print (folders)

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs = vgg.input, outputs = prediction)

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_data_gen = ImageDataGenerator(rescale = 1./255)

#트레인
train_set = train_data_gen.flow_from_directory('C:/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train',
                                               target_size = (224,224), batch_size = 32, class_mode = 'categorical')

#테스트 셋
test_set = test_data_gen.flow_from_directory('C:/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/valid',
                                             target_size = (224,224), batch_size = 32, class_mode = 'categorical')
import matplotlib.pyplot as plt
plt.imshow(plt.imread("/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/Tomato___Bacterial_spot/00416648-be6e-4bd4-bc8d-82f43f8a7240___GCREC_Bact.Sp 3110.JPG"))
plt.title("세균반점")
plt.show()

plt.imshow(plt.imread("/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/Tomato___Early_blight/0034a551-9512-44e5-ba6c-827f85ecc688___RS_Erly.B 9432.JPG"))
plt.title("겹무늬병")
plt.show()

plt.imshow(plt.imread("/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/Tomato___Late_blight/0003faa8-4b27-4c65-bf42-6d9e352ca1a5___RS_Late.B 4946.JPG"))
plt.title("감자역병균")
plt.show()

mod = model.fit_generator(
  train_set,
  validation_data=test_set,
  epochs=20,
  steps_per_epoch=len(train_set),
  validation_steps=len(test_set)
)
#매 에포크 마다 loss 훈련 손실값 , val_loss 검증 손실값
import matplotlib.pyplot as plt
plt.plot(mod.history['loss'], label='train loss')
plt.plot(mod.history['val_loss'], label='val loss')
plt.legend()
plt.show()
#accuracy 훈련 정확도 , val_accuracy 검증 정확도
plt.plot(mod.history['accuracy'], label='train accuracy')
plt.plot(mod.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

# save it as a h5 file
from tensorflow.keras.models import load_model
#model.save('model.h5')