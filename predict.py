import os
import pandas as pd
import torch
import warnings
from transformers import BertTokenizer, BertForSequenceClassification
import re

warnings.filterwarnings("ignore")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    models_dir = "models"
    data_dir = "data"
    mapping_file = os.path.join(data_dir, "label_mapping.csv")
    
    if not os.path.exists(models_dir) or not os.path.exists(mapping_file):
        print("!! Model or label mapping not found. Please run src/train_model.py first.")
        return
        
    mapping_df = pd.read_csv(mapping_file)
    id2label = {row['label_id']: row['label'] for _, row in mapping_df.iterrows()}
    
    print(f"Loading tokenizer and model from '{models_dir}'...")
    try:
        tokenizer = BertTokenizer.from_pretrained(models_dir)
        model = BertForSequenceClassification.from_pretrained(models_dir)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    model.eval()
    
    print("\n" + "="*50)
    print("        Medical Specialty Predictor API        ")
    print("="*50)
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            text = input(">> Enter a medical note: ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
            
        if text.strip().lower() == 'exit':
            print("Goodbye!")
            break
            
        if text.strip() == "":
            print("!! Please enter a valid medical note.")
            continue
            
        cleaned_txt = clean_text(text)
        if not cleaned_txt:
            print("!! Invalid input text. Could not process.")
            continue
            
        inputs = tokenizer(cleaned_txt, return_tensors="pt", truncation=True, padding=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_id = torch.argmax(logits, dim=1).item()
            confidence = probs[0][predicted_class_id].item()
            
        predicted_label = id2label[predicted_class_id]
        
        print("-" * 50)
        print(f"Predicted Specialty : {predicted_label}")
        print(f"Confidence Score    : {confidence:.2f}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
