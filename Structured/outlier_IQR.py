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


def remove_outliers_combined_IQR(df, columns):
    indices_to_keep = set(range(df.shape[0]))  # 모든 인덱스를 유지하는 집합 시작

    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 이상치가 아닌 데이터의 인덱스를 구함
        valid_indices = df.index[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        indices_to_keep.intersection_update(valid_indices)  # 유효한 인덱스만 유지

    # 이상치가 아닌 행만 포함하는 데이터프레임 반환
    filtered_df = df.loc[indices_to_keep]
    return filtered_df


df = create_example_dataset()

# 연속형 컬럼들에 대해 이상치 제거
continuous_columns = ['feature1', 'feature2', 'feature3']
cleaned_df = remove_outliers_combined_IQR(df, continuous_columns)
