import matplotlib
from keras.backend import expand_dims
from  numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
#plt 한글 폰트
matplotlib.rcParams['font.family'] ='Malgun Gothic'

matplotlib.rcParams['axes.unicode_minus'] =False
#plt 한글폰트 끝

#이미지
img = load_img ("/Users/ik533/Desktop/TomatoAug/train/Tomato___Bacterial_spot/0b13b997-9957-4029-b2a4-ef4a046eb088___UF.GRC_BS_Lab Leaf 0595.JPG")

print(img)
data = img_to_array(img)
print(data)
print(data.shape)
plt.imshow(plt.imread("/Users/ik533/Desktop/TomatoAug/train/Tomato___Bacterial_spot/0b13b997-9957-4029-b2a4-ef4a046eb088___UF.GRC_BS_Lab Leaf 0595.JPG"))
plt.title("세균반점")
plt.show()


samples =expand_dims(data,0)

datagen = ImageDataGenerator(width_shift_range=[-200,200])

it = datagen.flow(samples, batch_size=1)

fig = plt.figure(figsize=(30,30))

for i in range(9):
    plt.subplot(3,3, i+1)

    batch = it.next()

    image =batch[0].astype('uint8')

    plt.imshow(image)
    plt.show()