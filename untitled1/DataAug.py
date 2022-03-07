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

print("[INFO] loading example image...")
image = load_img(args["image"])
image = img_to_array(image)
image = np.expand_dims(image, axis=0)  # 맨 앞 1차원 추가

aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

print("[INFO] generating images...")
imageGen = aug.flow(
    image,
    batch_size=1,
    save_to_dir=args["output"],
    save_prefix=args["prefix"],
    save_format="jpg",
)

total = 0
# 그런 다음 imageGen 생성기의 각 이미지를 반복하기 시작합니다.
# 내부적으로 imageGen은 루프를 통해 요청 될 때마다 새로운 학습 샘플을 자동으로 생성합니다.
# 그런 다음 디스크에 기록 된 총 데이터 증가 예제 수를 늘리고
# 예제 10 개에 도달하면 스크립트 실행을 중지합니다.
# loop over examples from our image data augmentation generator
for image in imageGen:
    # increment our counter
    total += 1

    # if we have reached 10 examples, break from the loop
    if total == 10:
        break

#python augmentation.py -i jemma.png -o output -p image