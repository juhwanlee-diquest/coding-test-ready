from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 예제 데이터셋 생성
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


# 데이터 생성
df = create_example_dataset()

# 데이터 정규화 (DBSCAN 성능 향상을 위해)
X = StandardScaler().fit_transform(df[['feature1', 'feature2', 'feature3']])

# DBSCAN 모델 생성 및 학습
# eps: 이웃을 정의하는 최대 거리, min_samples: 핵심 포인트를 정의하기 위한 최소 이웃 수
db = DBSCAN(eps=0.5, min_samples=5).fit(X)

# 클러스터 라벨 할당 (-1 라벨은 이상치를 나타냄)
df['cluster'] = db.labels_

# 이상치 추출
outliers = df[df['cluster'] == -1]
normal_data = df[df['cluster'] != -1]

# 결과 출력
print("이상치 수:", outliers.shape[0])
print("정상 데이터 수:", normal_data.shape[0])