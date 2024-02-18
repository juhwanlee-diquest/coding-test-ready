import torch
import torch.nn as nn
import torch.nn.functional as F

# 딥 뉴럴 네트워크 정의
class SimpleDNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 모델 인스턴스 생성
input_size = 784 # 예: MNIST 이미지 데이터의 크기는 28x28
hidden_size = 500 # 은닉층의 노드 수
num_classes = 10 # 출력 클래스의 수 (예: MNIST의 경우 0부터 9까지의 숫자)
model = SimpleDNN(input_size, hidden_size, num_classes)

# 모델 요약 출력
print(model)
