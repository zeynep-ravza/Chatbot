import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_excel("veri seti")
df['text'] = df['text'].fillna('').astype(str)
df['expected_answer'] = df['expected_answer'].fillna('').astype(str)

# -----------------------------
# 2. TF-IDF Vektörleştirme
# -----------------------------
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'].tolist())

# -----------------------------
# 3. Top-k Retrieval Fonksiyonu
# -----------------------------
def retrieve_top_k_tf_idf(user_input, top_k=3):
    """
    Kullanıcı girişini TF-IDF vektörüne dönüştürür,
    cosine similarity ile top-k en uygun kayıtları döner.
    """
    user_vec = vectorizer.transform([user_input])
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    # En yüksek top-k benzerlik skorlarının indeksleri
    top_indices = similarities.argsort()[::-1][:top_k]

    # Top-k sonuçlarını liste olarak döndür
    results = []
    for idx in top_indices:
        results.append({
            "question": df.loc[idx, "text"],
            "answer": df.loc[idx, "expected_answer"],
            "similarity": float(similarities[idx])
        })
    return results

if __name__ == "__main__":
    print("Merhaba! Size nasıl yardımcı olabilirim? (Çıkmak için q yazın)")
    while True:
        query = input("\nKullanıcı: ").strip()
        if query.lower() in ["q", "quit", "exit", "çık"]:
            print("Görüşmek üzere, program kapatılıyor...")
            break

        retrieved = retrieve_top_k_tf_idf(query, top_k=3)
        print("\n--- Retrieval Sonuçları ---")
        for r in retrieved:
            print(f"Soru: {r['question']}")
            print(f"Cevap: {r['answer']}")
            print(f"Benzerlik Skoru: {r['similarity']:.3f}")
            print("-" * 40)
