import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Eğitimde kullandığın model yolu
MODEL_PATH = "./models/intent_model"
ENCODER_PATH = "./models/intent_encoder.pkl"
MODEL_NAME = "dbmdz/bert-base-turkish-cased"

# Model ve tokenizer yükle
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
intent_encoder = joblib.load(ENCODER_PATH)

def predict_intent(text: str) -> str:
    """Kullanıcı inputundan intent tahmini döner"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=-1).item()

    predicted_intent = intent_encoder.classes_[predicted_class_id]
    return predicted_intent
