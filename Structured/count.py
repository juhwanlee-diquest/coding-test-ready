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


# 
df = create_example_dataset()

# 각 컬럼별 discrete 여부 파악 및 discrete한 경우 count
column_details = {}
for column in df.columns:
    unique_values = df[column].unique()
    if len(unique_values) <= 10 or df[column].dtype == 'object' or df[column].dtype == 'bool':
        # 컬럼이 discrete한 것으로 간주
        column_details[column] = {
            'is_discrete': True,
            'count': df[column].value_counts().to_dict()
        }
    else:
        column_details[column] = {
            'is_discrete': False,
            'count': None
        }

print(column_details)
