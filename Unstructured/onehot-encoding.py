# 각 단어를 고유한 인덱스로 표현하는 방식입니다. 따라서 단어의 개수에 따라 벡터의 차원이 매우 커질 수 있습니다.
# 단어 간의 유사성이나 의미를 반영하지 않고, 단순히 해당 단어의 존재 여부만을 표현합니다.
# 단어 간의 관계를 고려하지 않기 때문에 단어 벡터 간의 유사성을 계산할 수 없습니다.
# 공간 효율성이 떨어지고, 희소한 벡터로 표현되기 때문에 연산량이 많고 학습 속도가 느릴 수 있습니다.

from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 범주형 데이터 예시 (4개의 범주)
categories = ['사과', '바나나', '딸기', '오렌지']

# OneHotEncoder 객체 생성
encoder = OneHotEncoder()

# 범주형 데이터를 정수로 인덱싱
categories_encoded = np.arange(len(categories)).reshape(-1, 1)

# One-Hot Encoding 수행
onehot_encoded = encoder.fit_transform(categories_encoded)

# 결과 출력
print("One-Hot Encoded 데이터:")
print(onehot_encoded.toarray())
