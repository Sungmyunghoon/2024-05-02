#학생들의 대학 순위 및 입학 여부에 대한 데이터를 사용하여 
#입학 여부를 예측하는 LSTM을 활용한 간단한 신경망을 구현

import numpy as np   #numpy
import tensorflow as tf #tensorflow
import pandas as pd #pandas 를 불러온다

data = pd.read_csv('gpascore.csv') #csv파일을 불러온다
data = data.dropna() #누락된 값이 있는 경우 삭제

x데이터 = []  #x데이터를 생성 x = [ [gre], [gpa], [rank] ] 로 구성
y데이터 = data['admit'].values #Y데이터를 생성 y = 입학 여부(0,1)

for i, rows in data.iterrows(): #반복문
    x데이터.append([rows['gre'], rows['gpa'], rows['rank']]) #반복적으로 데이터들을 삽입해주는 코드

model = tf.keras.models.Sequential([ # Sequential을 쓰면 신경만 레이어들을 쉽게 만들어줌
    tf.keras.layers.LSTM(64, input_shape=(None, 3)),  # LSTM 층 추가
    tf.keras.layers.Dense(128, activation='tanh'), #Dense 층 추가 = tanh로 계산
    tf.keras.layers.Dense(1, activation='sigmoid'), #Dense 층 추가 = sigmoid로 계산
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #모델을 컴파일

# 입력 데이터를 LSTM에 맞게 형태 변환
x데이터 = np.array(x데이터)
x데이터 = np.expand_dims(x데이터, axis=1)

model.fit(x데이터, np.array(y데이터), epochs=1000) # 모델을 epochs 만큼 학습

# 예측
예측값 = model.predict(np.array([[[610, 3.20, 2]]])) #모델 예측 모델 = [ 미국의 대학원 수학 자격시험 610점 , 학점 3.20 , 등급이 2등 정도 되는 대학교]
print(예측값)
