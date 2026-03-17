import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import warnings

warnings.filterwarnings("ignore")

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    data_dir = "data"
    cleaned_file = os.path.join(data_dir, "cleaned_data.csv")
    models_dir = "models"
    results_dir = "results"
    
    if not os.path.exists(cleaned_file):
        raise FileNotFoundError(f"Cleaned data not found. Please run src/preprocess.py first.")
        
    df = pd.read_csv(cleaned_file)
    texts = df['cleaned_text'].tolist()
    labels = df['label_id'].tolist()
    
    num_labels = len(set(labels))
    
    print("Splitting data into 80% train and 20% test sets...")
    
    try:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels)
    except ValueError:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42)
            
    # Optional subsampling
    # sample_size = len(texts) # adjust this if running on slow device
    print(f"Train size: {len(train_labels)}, Test size: {len(test_labels)}")
    
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    print("Tokenizing data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)
    
    train_dataset = MedicalDataset(train_encodings, train_labels)
    test_dataset = MedicalDataset(test_encodings, test_labels)
    
    print("Loading BERT base model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    
    os.makedirs(results_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(results_dir, 'checkpoints'),
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=0,
        weight_decay=0.01,
        logging_dir=os.path.join(results_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Starting Model Training. Please wait...")
    trainer.train()
    
    print("Evaluating Model on Test Set...")
    eval_results = trainer.evaluate()
    
    report_path = os.path.join(results_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Evaluation Report\n")
        f.write("=================\n")
        f.write(f"Accuracy : {eval_results.get('eval_accuracy', 0):.4f}\n")
        f.write(f"Precision: {eval_results.get('eval_precision', 0):.4f}\n")
        f.write(f"Recall   : {eval_results.get('eval_recall', 0):.4f}\n")
        f.write(f"F1 Score : {eval_results.get('eval_f1', 0):.4f}\n")
        
    print(f"\nEvaluation complete. Report saved to {report_path}")
    
    os.makedirs(models_dir, exist_ok=True)
    print("Saving final trained model...")
    model.save_pretrained(models_dir)
    tokenizer.save_pretrained(models_dir)
    print(f"--- Training Complete! Model saved to '{models_dir}' ---")

if __name__ == "__main__":
    main()
