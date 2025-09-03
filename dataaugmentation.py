import pandas as pd
from tqdm import tqdm
import nlpaug.augmenter.word as naw
from transformers import pipeline
import re

# Hugging Face paraphrase pipeline
paraphrase = pipeline("text2text-generation", model="ramsrigouthamg/t5_paraphraser")

# Synonym augmenter
syn_aug = naw.SynonymAug(aug_src='wordnet')

# --- TEMİZLEME FONKSİYONLARI ---

def clean_repeated_words(text):
    """Aynı kelimenin peş peşe tekrar etmesini engeller."""
    # Tekrar eden kelimeleri tekilleştir
    text = re.sub(r'\b(\w+)( \1){1,}\b', r'\1', text)
    # Çok fazla tekrar varsa kırp
    words = text.split()
    if len(words) > 30:  # çok uzun tekrarları engelle
        text = " ".join(words[:30])
    return text.strip()

def remove_bad_terms(text):
    """Bağlam dışı veya anlamsız kelimeleri temizler."""
    bad_terms = [
        "volt ampere", "myocardial infarct", "atomic number", "section 5",
        "knot", "logos", "neon"
    ]
    for term in bad_terms:
        text = text.replace(term, "")
    return text.strip()

def clean_text(text):
    """Hem tekrarları hem bağlam dışı kelimeleri temizler."""
    if not isinstance(text, str):
        return text
    text = clean_repeated_words(text)
    text = remove_bad_terms(text)
    return text

# --- ANA ÇALIŞMA ---

# Veri setini oku
df = pd.read_excel("C:/Users/ZEYNEP/Desktop/chatbot/data/veriseti.xlsx")
df = df.dropna(subset=['Soru / Mesaj İçeriği'])

new_rows = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    original_text = row['Soru / Mesaj İçeriği']
    base_id = str(row['ID'])

    # 1) Paraphrase üret
    para_result = paraphrase(f"paraphrase: {original_text}", max_length=70, do_sample=True)
    paraphrased_text = para_result[0]['generated_text']
    paraphrased_text = clean_text(paraphrased_text)  # Temizlik

    # 2) Synonym replacement
    synonym_text = syn_aug.augment(original_text)
    if isinstance(synonym_text, list):
        synonym_text = " ".join(synonym_text)  # liste ise stringe çevir
    synonym_text = clean_text(synonym_text)  # Temizlik

    # Paraphrased satır
    new_rows.append({
        'ID': f"{base_id}_p",
        'Soru / Mesaj İçeriği': paraphrased_text,
        'Kategori': row['Kategori'],
        'Intent': row['Intent'],
        'Duygu': row['Duygu'],
        'Beklenen Cevap': row['Beklenen Cevap'],
        'LİNK': row['LİNK']
    })

    # Synonym replaced satır
    new_rows.append({
        'ID': f"{base_id}_s",
        'Soru / Mesaj İçeriği': synonym_text,
        'Kategori': row['Kategori'],
        'Intent': row['Intent'],
        'Duygu': row['Duygu'],
        'Beklenen Cevap': row['Beklenen Cevap'],
        'LİNK': row['LİNK']
    })

# Orijinal + yeni satırlar birleştir
augmented_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# Excel dosyası olarak kaydet
augmented_df.to_excel("augmented_dataset_cleaned.xlsx", index=False, encoding="utf-8-sig")

print("✅ Veri augmentation + temizlik tamamlandı! Toplam satır sayısı:", len(augmented_df))
