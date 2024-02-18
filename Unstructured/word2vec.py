# 단어를 밀집(dense) 벡터로 표현하는 방식으로, 단어의 의미와 관계를 고려하여 벡터 공간 상에 임베딩합니다.
# 단어 벡터 간의 유사성을 계산할 수 있어서 단어 간의 의미적 관계를 파악하는 데 유용합니다.
# 벡터의 차원이 상대적으로 낮아 공간 효율성이 높고, 희소하지 않아서 연산량이 적고 학습 속도가 빠를 수 있습니다.
# 단어 임베딩을 학습하기 위해서는 대규모의 텍스트 데이터가 필요하며, Word2Vec 모델의 하이퍼파라미터를 조정해야 할 수 있습니다.

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')  # NLTK의 토크나이저를 사용하기 위해 필요

# 예시 텍스트 데이터
corpus = [
    "This is the first sentence for Word2Vec example.",
    "This is the second sentence.",
    "Yet another sentence is here.",
    "One more sentence follows.",
    "And the final sentence."
]

# 텍스트 데이터를 단어 단위로 토큰화
tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]

# Word2Vec 모델 학습
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# 단어 'sentence'의 벡터 표현 확인
print("Vector representation of 'sentence':", model.wv['sentence'])

# 유사한 단어 찾기
similar_words = model.wv.most_similar('sentence')
print("Words most similar to 'sentence':", similar_words)
