import torch
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from retrieval import retrieve_top_k_tf_idf  

# RAG modeli yükleme
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_rag_answer(user_query, top_k=3):
    retrieved_texts = retrieve_top_k_tf_idf(user_query, top_k) 
    
    if len(retrieved_texts) == 0:
        return "Bu konuda elimde yeterli bilgi bulunmamaktadır."
    
    context = " ".join(retrieved_texts)
    inputs = tokenizer([user_query], context=[context], return_tensors="pt").to(device)
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=200)
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    return answer

if __name__ == "__main__":
    print("Merhaba! Size nasıl yardımcı olabilirim? (Çıkmak için 'q')")
    while True:
        user_input = input("Kullanıcı: ").strip()
        if user_input.lower() in ["q", "quit", "exit", "çık"]:
            break
        response = generate_rag_answer(user_input)
        print(f"Bot: {response}\n")
