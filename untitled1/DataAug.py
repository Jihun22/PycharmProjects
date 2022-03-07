#데이터 증대 공부
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from imutils import paths
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
ap.add_argument("-o","--output", required=True)
ap.add_argument("-p","--prefix", type=str, default="image")
args = vars(ap.parse_args())


