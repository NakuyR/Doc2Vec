import pandas as pd
from konlpy.tag import Okt
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
from gensim.models import doc2vec

def doc2vec_learning(df):
    df=df[['요약','링크']]
    df = df.dropna() #NaN 값 제거

#데이터 전처리
    mecab = Okt()

    tagged_corpus_list = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        text = row['요약']
        tag = row['링크']
        tagged_corpus_list.append(TaggedDocument(tags=mecab.morphs(tag), words=mecab.morphs(text)))  # 데이터  토큰화하기

    model = doc2vec.Doc2Vec(vector_size=300, alpha=0.025, min_alpha=0.025, workers=8, window=8)

# Vocabulary 빌드
    model.build_vocab(tagged_corpus_list)

# Doc2Vec 학습
    model.train(tagged_corpus_list, total_examples=model.corpus_count, epochs=50)

# 모델 저장
    model.save('삼성전자.doc2vec')
    

def similar_doc2vec(df,url):
    doc2vec_learning(df)
    loaded_model = doc2vec.Doc2Vec.load('삼성전자.doc2vec')  #훈련한 모델 load
    similar_doc = loaded_model.docvecs.most_similar(url)
    return similar_doc