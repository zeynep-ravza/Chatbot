import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(filepath):
    """
    JSON veri setini yükler ve ML için hazırlar
    """
    # JSON dosyasını yükle
    df = pd.read_json(filepath, encoding="utf-8")
    
    # Text kolonunu ayarla
    if 'Soru / Mesaj İçeriği' in df.columns:
        df['text'] = df['Soru / Mesaj İçeriği']
    else:
        raise ValueError("❌ 'Soru / Mesaj İçeriği' kolonu JSON dosyasında yok!")

    # Beklenen cevap kolonunu ayarla
    if 'Beklenen Cevap' in df.columns:
        df['expected_answer'] = df['Beklenen Cevap']
    else:
        df['expected_answer'] = ""

    # NaN değerleri doldur
    df['text'] = df['text'].fillna('')
    df['expected_answer'] = df['expected_answer'].fillna('')
    
    # Intent encoding
    if 'Intent' not in df.columns:
        raise ValueError("❌ JSON dosyasında 'Intent' kolonu yok!")
    
    intent_encoder = LabelEncoder()
    df['intent_label'] = intent_encoder.fit_transform(df['Intent'].astype(str))

    return df, intent_encoder
