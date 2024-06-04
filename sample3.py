import pandas as pd
from konlpy.tag import Komoran
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# 데이터 전처리
def extract_nouns(text):
    # 형태소 분석기 초기화
    komoran = Komoran()

    # 명사만 추출
    nouns = komoran.nouns(text)
    
    return ' '.join(nouns)

#글 요약
def text_rank_summary(text, num_sentences=1):
    # 문장 토큰화
    sentences = sent_tokenize(text)

    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    words = [word for word in words if word not in stop_words]

    # 단어 빈도수 계산
    word_freq = FreqDist(words)

    # 문장 가중치 계산 (TextRank 알고리즘)
    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        sentence_score = sum([word_freq[word] for word in sentence_words if word in word_freq])
        sentence_scores[sentence] = sentence_score

    # 문장 중요도 순으로 정렬
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

    # 상위 N개의 문장 선택하여 요약문 생성
    summary = [sentence for sentence, _ in sorted_sentences[:num_sentences]]

    return ' '.join(summary)

#단어 추출
def extract_keywords(text, num_keywords=3):
    # 명사만 추출
    nouns_text = extract_nouns(text)

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([nouns_text])

    # 단어와 인덱스 매핑
    word2index = {word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())}

    # TF-IDF 값이 큰 순서대로 정렬하여 상위 N개 단어 선택
    sorted_indices = np.argsort(tfidf_matrix.toarray()[0])[::-1][:num_keywords]
    keywords = [vectorizer.get_feature_names_out()[idx] for idx in sorted_indices]

    return keywords

def main():
    # 사용자로부터 입력 받기
    user_input = input("장문의 글을 입력하세요: ")

    # 요약
    summary = text_rank_summary(user_input)

    # 키워드 추출
    keywords = extract_keywords(user_input)

    # 요약문과 키워드를 CSV 파일에 저장
    data = {'Summary': [summary], 'Keywords': [', '.join(keywords)]}
    df = pd.DataFrame(data)
    df.to_csv('summary_and_keywords.csv', index=False)

    print("\n요약문:")
    print(summary)
    print("\n추출된 키워드:")
    print(', '.join(keywords))
    print("\nCSV 파일에 저장되었습니다.")

if __name__ == "__main__":
    main()
