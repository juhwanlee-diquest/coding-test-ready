from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import numpy as np

# 예제 데이터셋 생성
def create_example_dataset(num_samples=1000):
    np.random.seed(0)  # 결과의 재현성을 위해 난수 시드 설정
    data = {
        'feature1': np.random.rand(num_samples),
        'feature2': np.random.rand(num_samples),
        'feature3': np.random.rand(num_samples),
        'feature4': np.random.rand(num_samples),
        'target': np.random.randint(0, 2, num_samples)  # 이진 분류 문제를 위한 타겟 변수 (0 또는 1)
    }
    df = pd.DataFrame(data)
    return df


# 데이터 생성
df = create_example_dataset()


# LDA 초기화 (이진 분류 문제이므로, 최대 축소 가능 차원은 1)
lda = LinearDiscriminantAnalysis(n_components=1)


# 축소
features = ['feature1', 'feature2', 'feature3', 'feature4']
X = df[features]
y = df['target']

X_lda = lda.fit_transform(X, y)

lda_df = pd.DataFrame(X_lda, columns=['LDA1'])


print(lda_df)