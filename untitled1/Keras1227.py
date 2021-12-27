from keras.datasets import boston_housing
# 보스턴  주택 데이터셋 로드
(train_data, train_targets),(test_data, test_targets) = boston_housing.load_data()
print(train_data.shape)
print(test_data.shape)
print(train_targets)

#데이터 준비
#데이터 정규화
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
