from sklearn.manifold import TSNE
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


# tsne 모델 초기화 (2차원으로 축소)
tsne = TSNE(n_components=2, random_state=0)


# 축소
features = ['feature1', 'feature2', 'feature3']
X = df[features]
X_pca = tsne.fit_transform(X)
tsne_df = pd.DataFrame(X_pca, columns=['TSNE1', 'TSNE2'])

print(tsne_df)