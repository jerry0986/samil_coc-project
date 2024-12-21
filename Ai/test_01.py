import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

# 데이터 경로 설정
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# 이미지 분류 모델 구축
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 이진 분류
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 데이터 전처리 및 모델 학습
def train_model():
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    model = build_model()
    model.fit(train_generator, epochs=10)
    model.save('weather_model.h5')  # 모델 저장

# 모델 로드
def load_model():
    return tf.keras.models.load_model('weather_model.h5')

# 예측 함수
def predict_weather(image_path):
    img = image.load_img(image_path, target_size=(150, 150))  # 이미지 크기 조정
    img_array = image.img_to_array(img) / 255.0  # 정규화
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = weather_model.predict(img_array)
    probability = prediction[0][0]  # 확률 값
    return probability

# 모델 학습
if not os.path.exists('weather_model.h5'):
    train_model()

# 모델 로드
weather_model = load_model()

# 반복 실행
while True:
    print("\n1: 기상 상황 이미지 예측")
    print("2: 종료")
    choice = input("원하는 작업의 번호를 입력하세요: ")

    if choice == '1':
        # 사용자 입력 이미지 경로
        image_path = input("예측할 이미지 경로를 입력하세요: ")
        if os.path.exists(image_path):
            probability = predict_weather(image_path)
            if probability > 0.5:
                print(f"기상 변화가 있는 상황입니다. 확률: {probability * 100:.2f}%")
            else:
                print(f"평소 상황입니다. 확률: {probability * 100:.2f}%")
        else:
            print("이미지 경로가 존재하지 않습니다.")

    elif choice == '2':
        print("프로그램을 종료합니다.")
        break

    else:
        print("유효하지 않은 선택입니다. 다시 시도하세요.")
