import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_excel("veri seti")



# NaN değerleri temizle
df['text'] = df['text'].fillna('').astype(str)
df['expected_answer'] = df['expected_answer'].fillna('').astype(str)

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'].tolist())

def retrieve_best_answer(user_input):
    """
    Kullanıcı girişini alır, TF-IDF vektörüne dönüştürür ve
    cosine similarity ile en uygun cevabı döner.
    """
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx = similarities.argmax()

    best_answer = df.loc[best_idx, "expected_answer"]
    return best_answer


if __name__ == "__main__":
    print("Merhaba! Size nasıl yardımcı olabilirim? (çıkmak için q yazın)")
    while True:
        query = input("\nKullanıcı: ").strip()
        if query.lower() in ["q", "quit", "exit", "çık"]:
            print("Görüşmek üzere, program kapatılıyor...")
            break

        answer = retrieve_best_answer(query)
        print(f"Bot: {answer}")
