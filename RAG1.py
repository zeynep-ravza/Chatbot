import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve(user_question_input):
    with open("veriseti.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Boş ve sayısal olmayan satırları temizle
    cleaned_data = []
    for item in data:
        soru = item.get('Soru / Mesaj İçeriği')
        cevap = item.get('Beklenen Cevap')
        if isinstance(soru, str) and isinstance(cevap, str) and soru.strip() and cevap.strip():
            cleaned_data.append(item)

    knowledge_base = [item['Soru / Mesaj İçeriği'] for item in cleaned_data]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(knowledge_base)

    user_vec = vectorizer.transform([user_question_input])
    similarity_scores = cosine_similarity(user_vec, tfidf_matrix)
    best_match_index = np.argmax(similarity_scores)
    best_match_score = similarity_scores[0, best_match_index]

    best_match_question = knowledge_base[best_match_index]
    best_match_answer = cleaned_data[best_match_index]['Beklenen Cevap']
    best_match_link = cleaned_data[best_match_index]['LİNK']

    return user_question_input, best_match_answer, best_match_link, best_match_score
