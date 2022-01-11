import numpy as np
import h5py
import matplotlib.pyplot as plt

# 데이터 세트 로드 함수
def load_dataset():
    # h5py 모듈 사용 및, 파일 경로와 파일 접속 모드(읽기, read)를 설정
    all_train_data = h5py.File('C:/Users/ik533/Desktop/h5py/train_happy.h5', "r")
    all_test_data = h5py.File('C:/Users/ik533/Desktop/h5py/test_happy.h5', "r")

    # 모든 훈련 데이터와 테스트 데이터를 파일에서 읽어 numpy 배열에 저장
    x_train = np.array(all_train_data["train_set_x"][:])
    y_train = np.array(all_train_data["train_set_y"][:])

    x_test = np.array(all_test_data["test_set_x"][:])
    y_test = np.array(all_test_data["test_set_y"][:])

    # 데이터 셰이프 변경
    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    return x_train, y_train, x_test, y_test
# 데이터 로드
X_train, Y_train, X_test, Y_test = load_dataset()

print('Image dimension: ', X_train.shape[1:])
print('Training tensor dimension: ', X_train.shape)
print('Test tensor dimention: ', X_test.shape)
print()
print('Number of training tensor: ', X_train.shape[0])
print('Number of test tensor: ', X_test.shape[0])
print()
print('Training lable dimension: ', Y_train.shape)
print('Test label dimension 차원: ', Y_test.shape)

# 이미지 플롯
plt.imshow(X_train[0])

# 이미지 라벨 출력(smiling = 1, frowning = 0)
print("y = " + str(np.squeeze(Y_train[:, 0])))