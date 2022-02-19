import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
from glob import glob
# Model / data parameters

#데이터 경로

path = '/Users/ik533/Desktop/archive/New Plant Diseases Dataset(Augmented)'
os.listdir(path)

print (path)

#훈련 , 테스트 경로
train_path = os.path.join(path, "train")
print(os.listdir(train_path))
print("*"*100)
test_path = os.path.join(path, "valid")
print(os.listdir(test_path))



