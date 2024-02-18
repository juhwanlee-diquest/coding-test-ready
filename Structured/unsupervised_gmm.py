import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
# import matplotlib.pyplot as plt

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

# GMM 모델 초기화 및 학습
gmm = GaussianMixture(n_components=2, random_state=0)  # 2개의 가우시안 분포를 가정
gmm.fit(df[['feature1', 'feature2', 'feature3']])

# 클러스터 할당
clusters = gmm.predict(df[['feature1', 'feature2', 'feature3']])

# # 클러스터 시각화
# plt.scatter(df['feature1'], df['feature2'], c=clusters, cmap='viridis')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('GMM Clustering')
# plt.colorbar(label='Cluster')
# plt.show()
