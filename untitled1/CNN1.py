import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
plt.show()
sess = tf.Session()

#1 도형 인식 데이터셋 구성하기
x_data = []
y_data = []
im =10

for i in range(40) :
    img = Image.open(('shapedata/네모%d.png') % (i+1))
    gray = img.convert('L').resize((im,im))
    data = np.array(gray , dtype='uint8')
    x_data.append (  data.flatten())
    y_data.append(0)

for i in range(40):
    img = Image.open(('shapedata/세모%d.png')% (i+1))
    gray = img.convert('L').resize((im,im))
    data = np.array( gray , dtype='uint8')
    x_data.append( data.flatten() )
    y_data.append(1)

for i in range(40):
    img = Image.open(('shapedata/원%d.png')% (i+1))
    gray = img.convert('L').resize((im,im))
    data = np.array( gray , dtype='uint8')
    x_data.append( data.flatten() )
    y_data.append(2)

x_data = np.array(x_data) / 255.0
y_data = np.array(y_data)
print(x_data.shape)
print(y_data.shape)