import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

# CSV 파일에서 데이터 읽기
data = pd.read_csv(r'C:\Users\MIL-07\Desktop\python\dataset\tp.csv')

# 풍속 데이터 추출
wind_speed = data['ms'].tolist()  # 풍속 (m/s)

# 풍속의 평균값 계산
average_wind_speed = sum(wind_speed) / len(wind_speed)
print(f"풍속의 평균값: {average_wind_speed:.2f} m/s")

# 데이터프레임 생성
data = pd.DataFrame({'m/s': wind_speed})

# 태풍 발생 여부 생성
data['태풍발생여부'] = (data['m/s'] > 0).astype(int)

# 풍속과 태풍 발생 여부 사용
X = data[['m/s']]
y = data['태풍발생여부']

# 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# 모델 정확도 확인
y_pred = model.predict(X_test)
print(f'모델 정확도: {accuracy_score(y_test, y_pred)}')

# 사용자 입력
current_wind_speed = float(input("현재 풍속을 입력하세요 (단위: m/s): "))
current_temperature = float(input("현재 온도를 입력하세요 (단위: °C): "))

# 입력 데이터를 DataFrame으로 변환
input_data = pd.DataFrame([[current_wind_speed]], columns=['m/s'])

# 예측 확률 계산
prediction_proba = model.predict_proba(input_data)  # 예측 확률
storm_probability = prediction_proba[0][1]  # 태풍 발생 확률

# 온도 조건 반영
if 15 <= current_temperature <= 27:
    if current_wind_speed < average_wind_speed:
        final_probability = storm_probability * 0.5  # 풍속이 평균값보다 낮을 경우 확률 감소
    else:
        final_probability = storm_probability  # 풍속이 평균값 이상일 경우
else:
    final_probability = storm_probability * (current_temperature / 27)

# 이미지 비교를 위한 함수들
def load_image(image_path):
    """이미지를 로드하고 그레이스케일로 변환합니다."""
    print(f"로드할 이미지 경로: {image_path}")  # 경로 출력
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def compare_images(imageA, imageB):
    """두 이미지를 비교하고 SSIM을 계산합니다."""
    score, diff = ssim(imageA, imageB, full=True)
    diff = (diff * 255).astype("uint8")  # 차이 이미지를 0-255 범위로 변환
    return score, diff

def display_results(imageA, imageB, diff, score):
    """결과를 출력합니다."""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("input image")
    plt.imshow(imageA, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("most similar image")
    plt.imshow(imageB, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Difference (SSIM: {score:.2f})\nWind Speed: {current_wind_speed} m/s\nTemperature: {current_temperature} °C")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 이미지가 저장된 디렉토리 경로
directory = 'C:/Users/MIL-07/Desktop/python/xovnd'

# 디렉토리 내의 모든 파일을 순회하여 이미지 파일 목록 생성
image_files = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

# 사용자에게 이미지 경로 입력 받기
image_pathA = input("이미지의 경로를 입력하세요: ")

if image_pathA:
    print(f"선택한 이미지 경로: {image_pathA}")
else:
    print("이미지가 선택되지 않았습니다.")

def find_most_similar_image(target_image_path, directory):
    target_image = load_image(target_image_path)
    best_score = -1
    best_image_path = None

    # 디렉토리 내의 모든 이미지 파일을 비교
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            current_image = load_image(image_path)

            # 이미지 크기 조정 (필요한 경우)
            if target_image.shape != current_image.shape:
                current_image = cv2.resize(current_image, (target_image.shape[1], target_image.shape[0]))

            score = compare_images(target_image, current_image)[0]  # SSIM 점수만 가져옴

            print(f"Comparing with {filename}: SSIM = {score}")

            if score > best_score:
                best_score = score
                best_image_path = image_path

    return best_image_path, best_score

# 이미지 비교 실행
if image_pathA and image_files:
    most_similar_image, similarity_score = find_most_similar_image(image_pathA, directory)

    if most_similar_image:
        print(f"가장 유사한 사진: {most_similar_image} (SSIM: {similarity_score})")
        
        # 결과를 시각화하기 위해 두 이미지를 로드
        imageA = load_image(image_pathA)
        imageB = load_image(most_similar_image)

        # 이미지 크기 조정 (필요한 경우)
        if imageA.shape != imageB.shape:
            imageB = cv2.resize(imageB, (imageA.shape[1], imageA.shape[0]))

        score, diff = compare_images(imageA, imageB)
        display_results(imageA, imageB, diff, score)
    else:
        print("유사한 사진을 찾을 수 없습니다.")
else:
    print("A 이미지를 선택하지 않았거나, 디렉토리에 이미지 파일이 없습니다.")

# SSIM이 0.5 이하일 경우 SSIM 0으로 설정
if similarity_score <= 0.1:
    similarity_score = 0
else:
    similarity_score = similarity_score

if similarity_score >= 0.3 and similarity_score <= 0.5:
    similarity_score += 0.5 

# 각 요소의 비율
wind_speed_percentage = 15  # 풍속 비율
temperature_percentage = 5  # 온도 비율
image_similarity_percentage = 80  # 이미지 유사도 비율

# 최종 태풍 가능성 계산
if current_wind_speed > average_wind_speed  or current_temperature > 27:  # 평균값 초과 조건
    final_storm_probability = (final_probability * (wind_speed_percentage / 100)) + \
                              (similarity_score * (image_similarity_percentage / 100)) + \
                              ((current_temperature / 27) * (temperature_percentage / 100))
else:
    final_storm_probability = 0  # 조건을 만족하지 않으면 0으로 설정

# 최종 태풍 가능성 출력
print(f"최종 태풍 가능성: {final_storm_probability * 100:.2f}%")
