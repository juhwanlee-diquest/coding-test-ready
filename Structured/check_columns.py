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



# 각 컬럼별 discrete 여부 파악 및 discrete한 경우 count
def find_count_discrete_columns(df):    
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
    return column_details


# correlation between columns
def correlation(df, threshold=0.5):
    # 데이터프레임의 컬럼 간 상관관계 계산
    correlation_matrix = df.corr()

    # 상관관계의 절대값을 기준으로 높은 관계성을 가진 컬럼 쌍 찾기
    # 상관계수의 절대값이 (초기값 0.5) 이상인 경우를 '높은' 관계성으로 간주
    high_correlation_pairs = correlation_matrix.unstack().sort_values(ascending=False)
    high_correlation_pairs = high_correlation_pairs[(high_correlation_pairs < 1) & (abs(high_correlation_pairs) >=threshold)]

    return correlation_matrix, high_correlation_pairs


# 
df = create_example_dataset()

# find_count_discrete_columns
columns_info=find_count_discrete_columns(df)
print("Column Info\n: ", columns_info)

# correlation
correlation_matrix, high_correlation_pairs = correlation(df)
print('matrix:\n',correlation_matrix)
print("Correlation:\n", high_correlation_pairs)

