import torch
import cv2
import csv
import glob
import pandas as pd
from konlpy.tag import Komoran
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# 이미지 객체 인식 함수
def image_object_detection(image_folder, csv_filename):
    weights_path = "D:/yolov5/runs/train/exp_human/weights/human.pt"
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, force_reload=True)

    image_files = glob.glob(image_folder)

    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = ["Image", "Class", "Count", "Weight_Accuracy"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for image_path in image_files:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = model(img)

            human_count = (results.pred[0][:, -1] == 0).sum().item()

            # 수정: weight_accuracy 정보 얻는 방식에 따라 변경해야 함
            # weight_accuracy = results.model.model[-1].float()  # 이전 방식
            weight_accuracy = ...  # 수정 필요

            writer.writerow({"Image": image_path, "Class": "human", "Count": human_count, "Weight_Accuracy": weight_accuracy})

            print(f"Processed {image_path}: Detected 'human' objects: {human_count}")

    print(f"Result saved to {csv_filename}")

# 자연어 처리 함수
def natural_language_processing(user_input, min_word_count=3):
    def extract_nouns(text):
        komoran = Komoran()
        nouns = komoran.nouns(text)
        return ' '.join(nouns)

    def text_rank_summary(text, num_sentences=1):
        sentences = sent_tokenize(text)

        stop_words = set(stopwords.words('english'))
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalpha()]
        words = [word for word in words if word not in stop_words]

        word_freq = FreqDist(words)

        sentence_scores = {}
        for sentence in sentences:
            sentence_words = word_tokenize(sentence.lower())
            sentence_score = sum([word_freq[word] for word in sentence_words if word in word_freq])
            sentence_scores[sentence] = sentence_score

        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

        summary = [sentence for sentence, _ in sorted_sentences[:num_sentences]]

        return ' '.join(summary)

    def extract_keywords(text, num_keywords=3):
        nouns_text = extract_nouns(text)

        if len(nouns_text.split()) < min_word_count:
            raise ValueError(f"Input text must contain at least {min_word_count} words.")

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([nouns_text])

        word2index = {word: idx for idx, word in enumerate(vectorizer.get_feature_names_out())}

        sorted_indices = np.argsort(tfidf_matrix.toarray()[0])[::-1][:num_keywords]
        keywords = [vectorizer.get_feature_names_out()[idx] for idx in sorted_indices]

        return keywords

    summary = text_rank_summary(user_input)
    keywords = extract_keywords(user_input)

    data = {'Summary': [summary], 'Keywords': [', '.join(keywords)]}
    df = pd.DataFrame(data)

    return df

# 코사인 유사도 계산 함수
def calculate_cosine_similarity(sentence1, sentence2):
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    inputs = tokenizer([sentence1, sentence2], return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    sentence1_embedding = outputs.last_hidden_state[0][0].numpy()
    sentence2_embedding = outputs.last_hidden_state[0][1].numpy()
    similarity = cosine_similarity([sentence1_embedding], [sentence2_embedding])

    return similarity[0][0]

def main():
    # 이미지 객체 인식
    image_folder = "C:/Users/user/Desktop/yolov5/image/1.jpg"
    csv_filename = "C:/Users/user/Desktop/yolov5/human_result.csv"
    image_object_detection(image_folder, csv_filename)

    # 자연어 처리
    user_input = input("장문의 글을 입력하세요: ")
    processed_df = natural_language_processing(user_input, min_word_count=5)

    # 결과를 새로운 csv 파일에 저장
    processed_csv_file = "C:/Users/user/Desktop/yolov5/processed_result.csv"
    processed_df.to_csv(processed_csv_file, index=False)

    # 이미지 객체 인식 결과와 자연어 처리 결과를 병합하여 코사인 유사도 계산
    image_df = pd.read_csv(csv_filename)
    combined_df = pd.concat([image_df, processed_df], axis=1)

    # NaN 값을 빈 문자열로 대체
    combined_df['Summary'] = combined_df['Summary'].fillna('')

    # 한 줄 요약 추가 (Summary가 NaN인 경우 빈 문자열로 설정)
    combined_df['One_Line_Summary'] = combined_df['Summary'].apply(lambda x: " ".join(sent_tokenize(x)[:1]) if x else '')

    # 코사인 유사도 계산 함수에서 빈 문자열 또는 NaN 값인 경우에는 0으로 설정
    combined_df['Cosine_Similarity'] = combined_df.apply(lambda row: calculate_cosine_similarity(str(row['Summary']), str(row['One_Line_Summary']))
                                                        if row['Summary'] else 0, axis=1)

    # 코사인 유사도가 0.3 이상인 경우 키워드를 이용하여 자동으로 하나의 문장으로 요약
    combined_df['Combined_Sentence'] = combined_df.apply(lambda row: row['Summary']
                                                        if row['Cosine_Similarity'] < 0.3
                                                        else row['One_Line_Summary'], axis=1)
    # 결과를 새로운 csv 파일에 저장
    combined_csv_file = "C:/Users/user/Desktop/yolov5/combined_result.csv"
    combined_df[['Cosine_Similarity', 'Combined_Sentence']].to_csv(combined_csv_file, index=False)

    print(f"Combined sentences and saved to {combined_csv_file}")

if __name__ == "__main__":
    main()
