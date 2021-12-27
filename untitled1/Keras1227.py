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

#모델 구성
#모델 정의
from keras import  models
from keras import  layers

def build_model():
    #동일한 모델을 여러번 생성할 것이므로 함수를 만들어 사용
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return  model

# K-겹 검증을 사용한 훈련검증
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs =100
all_scores = []
for i in range(k):
    print('처리중인 폴드 #' , i)
    val_data =train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i +1) * num_val_samples]

    #훈련 데이터 준비 : 다른 분할 전체
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis =0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis = 0)

    # 케라스 모델 구성 (컴파일 포함)
    model = build_model()
    # 모델 훈련(verbose=0 이므로 훈련 과정이 출력되지 않습니다)
    model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs , batch_size=1, verbose =0)
    # 검증 세트로 모델 평가
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose =0)
    all_scores.append(val_mae)
print('all_scores:',all_scores)

print('np.mean:',np.mean(all_scores))

#각 폴드에서 검증 점수를 로그에 저장하기 500 포크 훈련
from keras import backend as K
# 메모리 해제
K.clear_session()
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('처리중인 폴드 #', i)
    # 검증 데이터 준비 :k 번째 분할
    val_data = train_data[i * num_val_samples: (i +1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    #훈련 데이터 준비 : 다른 분할 전체
    partial_train_data = np.concatenate (
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis= 0)
    # 케라스 모델 구성(컴파일 포함)
    model = build_model()
    # 모델 훈련 (verbose=0 이므로 훈련과정이 출력 되지 않습니다.)
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=0)
    mae_history = history.histroy['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
