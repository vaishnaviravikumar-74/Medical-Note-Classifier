import pandas as pd
import re
import os

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess():
    data_dir = "data"
    input_file = os.path.join(data_dir, "mtsamples.csv")
    cleaned_file = os.path.join(data_dir, "cleaned_data.csv")
    mapping_file = os.path.join(data_dir, "label_mapping.csv")
    
    print(f"Loading raw data from {input_file}...")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}")
        
    df = pd.read_csv(input_file)
    text_col = 'transcription' if 'transcription' in df.columns else df.columns[1]
    label_col = 'medical_specialty' if 'medical_specialty' in df.columns else df.columns[0]
    
    df = df.dropna(subset=[text_col, label_col])
    
    print("Cleaning text...")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    df = df[df['cleaned_text'].str.strip() != '']
    
    unique_labels = sorted(df[label_col].unique().tolist())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    
    df['label_id'] = df[label_col].map(label2id)
    
    # Save cleaned data
    df_cleaned = df[['cleaned_text', 'label_id', label_col]]
    df_cleaned.to_csv(cleaned_file, index=False)
    print(f"Cleaned data saved to {cleaned_file}")
    
    # Save label mapping
    mapping_df = pd.DataFrame(list(label2id.items()), columns=['label', 'label_id'])
    mapping_df.to_csv(mapping_file, index=False)
    print(f"Label mapping saved to {mapping_file}")

if __name__ == "__main__":
    preprocess()
