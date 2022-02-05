from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

image_size = [224, 224]

vgg = VGG16(input_shape = image_size + [3], weights = 'imagenet', include_top =  False)

for layer in vgg.layers:
    layer.trainable = False

from glob import glob
#데이터셋 위치
folders = glob('/User/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/*')

print (folders)

x = Flatten()(vgg.output)

prediction = Dense(len(folders), activation = 'softmax')(x)

model = Model(inputs = vgg.input, outputs = prediction)

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_gen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

test_data_gen = ImageDataGenerator(rescale = 1./255)

train_set = train_data_gen.flow_from_directory('/User/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/train/',
                                               target_size = (224,224), batch_size = 32, class_mode = 'categorical')

test_set = test_data_gen.flow_from_directory('/User/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)/valid/',
                                             target_size = (224,224), batch_size = 32, class_mode = 'categorical')
