import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd

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


class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(True)
        )
        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

X = df.drop('target', axis=1)


# 입력 데이터셋의 특성 수
input_dim = X.shape[1]
# 저차원 공간의 차원 수
encoding_dim = 2

# 모델 초기화
autoencoder = Autoencoder(input_dim, encoding_dim)

# 손실 함수 및 최적화 기법 정의
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

# 데이터를 PyTorch 텐서로 변환
X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(X.values, dtype=torch.float32)  # 오토인코더는 입력을 타겟으로 사용

# DataLoader 생성
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

# 모델 학습
num_epochs = 50
for epoch in range(num_epochs):
    for data in dataloader:
        inputs, targets = data
        # 순전파
        outputs = autoencoder(inputs)
        loss = criterion(outputs, targets)
        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 인코더를 사용한 차원 축소
encoded_samples = autoencoder.encoder(X_tensor).detach().numpy()

# 차원 축소 결과 확인
print(encoded_samples[:5])  # 처음 5개 샘플의 축소된 차원 출력
