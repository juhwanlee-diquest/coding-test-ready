from transformers import BertTokenizer, BertModel
import torch

# 사전 훈련된 모델과 토크나이저 초기화
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 인코딩 및 토큰 타입 아이디 생성
text = "Here is some text to encode"
encoded_input = tokenizer(text, return_tensors='pt')

# 모델을 통해 특성 추출
with torch.no_grad():
    output = model(**encoded_input)

# 첫 번째 토큰의 마지막 레이어 특성을 사용
features = output.last_hidden_state[:,0,:].numpy()
