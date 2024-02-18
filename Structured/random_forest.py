from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# 예제 데이터셋 생성 함수
def create_example_dataset(num_samples=1000):
    np.random.seed(0)  # 결과의 재현성을 위해 난수 시드 설정
    data = {
        'feature1': np.random.rand(num_samples),
        'feature2': np.random.rand(num_samples),
        'feature3': np.random.rand(num_samples),
        'target': np.random.randint(0, 2, num_samples)  # 이진 분류 문제를 위한 타겟 변수 (0 또는 1)
    }
    df = pd.DataFrame(data)
    return df

# 데이터셋 생성
df = create_example_dataset()

# 데이터셋 분할: 특성(features)과 타겟(target) 분리
X = df.drop('target', axis=1)  # 타겟 열 제외
y = df['target']  # 타겟 변수

# 훈련 세트와 테스트 세트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 랜덤 포레스트 모델 생성 및 훈련
random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_classifier.fit(X_train, y_train)

# 테스트 데이터에 대한 예측 수행
predictions = random_forest_classifier.predict(X_test)

# 정확도 평가
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
