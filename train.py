from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from src.data_loader import load_and_prepare_data
import numpy as np
import evaluate  

def tokenize_function(batch, tokenizer):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=256)

def main():
    df, intent_encoder, sentiment_encoder = load_and_prepare_data('veriseti')

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_pandas(train_df[['text', 'intent_label']])
    val_dataset = Dataset.from_pandas(val_df[['text', 'intent_label']])

    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_dataset = train_dataset.rename_column("intent_label", "labels")
    val_dataset = val_dataset.rename_column("intent_label", "labels")

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    num_labels = len(intent_encoder.classes_)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='./models/intent_model',
        eval_strategy="epoch",          # Burada parametre güncellendi
        save_strategy="epoch",          # Kaydetme stratejisi eklendi
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",  # eval_ prefix eklendi
        logging_dir='./logs',
        logging_steps=10,
    )

    metric = evaluate.load("accuracy")  # yeni evaluate paketi ile yüklendi

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model('./models/intent_model')

if __name__ == "__main__":
    main()
