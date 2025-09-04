from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from data_loader import load_and_prepare_data
import numpy as np
import evaluate  
import joblib  # Intent encoder'ı kaydetmek için

def tokenize_function(batch, tokenizer):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=256)

def main():
    # Veri yükle
    df, intent_encoder = load_and_prepare_data('veriseti.json')
    
    # Intent encoder'ı kaydet
    joblib.dump(intent_encoder, './models/intent_encoder.pkl')
    
    # Train/validation ayrımı
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Dataset oluştur
    train_dataset = Dataset.from_pandas(train_df[['text', 'intent_label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'intent_label']])

    # Model ve tokenizer yükle
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Kolon isimlerini düzenle
    train_dataset = train_dataset.rename_column("intent_label", "labels")
    val_dataset = val_dataset.rename_column("intent_label", "labels")

    # Format ayarla
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Model oluştur
    num_labels = len(intent_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./models/intent_model',
        eval_strategy="epoch",  # eski sürümde bu doğru
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        logging_dir='./logs',
        logging_steps=10,
    )

    # Metric
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Eğitimi başlat
    trainer.train()
    trainer.save_model("./models/intent_model")
    tokenizer.save_pretrained("./models/intent_model")

    
if __name__ == "__main__":
    import os
    os.makedirs('./models', exist_ok=True)
    main()
