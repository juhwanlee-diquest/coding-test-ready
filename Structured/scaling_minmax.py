from sklearn.preprocessing import MinMaxScaler
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

# MinMaxScaler 초기화
scaler = MinMaxScaler()

# 연속형 특성에 대해 MinMax Scaling 적용
scaled_features = scaler.fit_transform(df[['feature1', 'feature2', 'feature3']])

# 스케일링된 데이터를 DataFrame으로 변환
scaled_df = pd.DataFrame(scaled_features, columns=['feature1_scaled', 'feature2_scaled', 'feature3_scaled'])

# 스케일링된 특성을 원본 DataFrame에 추가
df_combined = pd.concat([df, scaled_df], axis=1)

print(df_combined)