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

# Define the weights paths for human.pt and food.pt
human_weights_path = "D:/yolov5/runs/train/exp_human/weights/human.pt"
food_weights_path = "D:/yolov5/runs/train/exp_food/weights/food.pt"

# 이미지 객체 인식 함수
def image_object_detection(image_folder, csv_filename, weights_path):
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

            if weights_path == human_weights_path:
                class_label = "human"
                object_count = (results.pred[0][:, -1] == 0).sum().item()
            else:
                class_label = "food"
                object_count = (results.pred[0][:, -1] == 0).sum().item()

            # 수정: weight_accuracy 정보 얻는 방식에 따라 변경해야 함
            # weight_accuracy = results.model.model[-1].float()  # 이전 방식
            weight_accuracy = ...  # 수정 필요

            writer.writerow({"Image": image_path, "Class": class_label, "Count": object_count, "Weight_Accuracy": weight_accuracy})

            print(f"Processed {image_path}: Detected '{class_label}' objects: {object_count}")

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

# Calculate cosine similarity using BERT embeddings
def calculate_cosine_similarity(text1, text2):
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Tokenize and encode the input texts
    inputs = tokenizer([text1, text2], padding=True, truncation=True, return_tensors='pt')

    # Get the embeddings
    with torch.no_grad():
        output = model(**inputs)
        embeddings = output.last_hidden_state

    # Calculate the cosine similarity if both texts are not empty
    if text1.strip() and text2.strip():
        similarity = cosine_similarity(embeddings[0], embeddings[1])
        return similarity
    else:
        return np.zeros((1, 1))  # Return a 1x1 zero matrix

def main():
    # 이미지 객체 인식 - human.pt
    human_image_folder = "C:/Users/user/Desktop/yolov5/image/*.jpg"
    human_csv_filename = "C:/Users/user/Desktop/yolov5/human_result.csv"
    image_object_detection(human_image_folder, human_csv_filename, human_weights_path)

    # 이미지 객체 인식 - food.pt
    food_image_folder = "C:/Users/user/Desktop/yolov5/image/*.jpg"
    food_csv_filename = "C:/Users/user/Desktop/yolov5/food_result.csv"
    image_object_detection(food_image_folder, food_csv_filename, food_weights_path)

    # 자연어 처리
    user_input = input("장문의 글을 입력하세요: ")
    processed_df = natural_language_processing(user_input, min_word_count=5)

    # 결과를 새로운 csv 파일에 저장
    processed_csv_file = "C:/Users/user/Desktop/yolov5/processed_result.csv"
    processed_df.to_csv(processed_csv_file, index=False)

    # 이미지 객체 인식 결과와 자연어 처리 결과를 병합하여 코사인 유사도 계산
    human_df = pd.read_csv(human_csv_filename)
    food_df = pd.read_csv(food_csv_filename)
    combined_df = pd.concat([human_df, food_df, processed_df], axis=1)

     # Create 'One_Line_Summary' column if it doesn't exist
    if 'One_Line_Summary' not in combined_df.columns:
        combined_df['One_Line_Summary'] = ''

    # Calculate the cosine similarity for each row
    combined_df['Cosine_Similarity'] = combined_df.apply(lambda row: calculate_cosine_similarity(str(row['Summary']), str(row['One_Line_Summary'])), axis=1)

    # Calculate the average cosine similarity for each row
    combined_df['Average_Cosine_Similarity'] = combined_df['Cosine_Similarity'].apply(lambda x: np.mean(x))

    # 코사인 유사도가 0.3 이상인 경우 키워드를 이용하여 자동으로 하나의 문장으로 요약
    def summarize(row):
        avg_similarity = row['Average_Cosine_Similarity']
        if avg_similarity >= 0.3:
            return row['Summary'] if not pd.isnull(row['Summary']) else ''  # Handle NaN
        else:
            summary = row['Summary'] if not pd.isnull(row['Summary']) else ''  # Handle NaN
            one_line_summary = row['One_Line_Summary'] if not pd.isnull(row['One_Line_Summary']) else ''  # Handle NaN
            return summary + ' ' + one_line_summary

    # Create the 'Combined_Sentence' column
    combined_df['Combined_Sentence'] = combined_df.apply(summarize, axis=1)

    # 결과를 새로운 csv 파일에 저장
    combined_csv_file = "C:/Users/user/Desktop/yolov5/combined_result.csv"
    combined_df[['Average_Cosine_Similarity', 'Combined_Sentence']].to_csv(combined_csv_file, index=False)

    print(f"Combined data and saved to {combined_csv_file}")

if __name__ == "__main__":
    main()
